import os
import re
import json
import time
import random
import pandas as pd
import mock_llm
from mock_llm import TransientLLMError, est_tokens

# ---------------- constants (top level: every function can read them) ----------------
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
CATEGORIES = ["billing", "delivery", "technical", "fraud"]

ZERO_SHOT_PROMPT = """You are a support ticket triage assistant.
CLASSIFY the customer ticket into exactly one of these categories:
- billing
- delivery
- technical
- fraud
Reply with JSON only, in exactly this form and nothing else:
{"category": "...", "reason": "..."}
The "category" value must be one of: billing, delivery, technical, fraud.
The "reason" value must be one short sentence.
Do not add any text, markdown or code fences outside the JSON."""

FEW_SHOT_PROMPT = ZERO_SHOT_PROMPT + """

EXAMPLES:
Ticket: I was charged twice for my order [ORDER_ID], please refund the extra amount.
Answer: {"category": "billing", "reason": "Customer reports a duplicate charge."}

Ticket: My parcel was supposed to arrive last week and the tracking has not moved.
Answer: {"category": "delivery", "reason": "Package is late and tracking is stuck."}

Ticket: The app crashes every time I open the payment page.
Answer: {"category": "technical", "reason": "App crashes, which is a software problem."}

Ticket: Someone used my card to place an order I never made, I did not authorise this.
Answer: {"category": "fraud", "reason": "Customer reports an unauthorised transaction."}"""


# ---------------- Task 1 + 2: guardrails and redaction ----------------
def guardrail_redact():
    df = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    email_count = phone_count = order_count = 0
    redacted_texts = []
    for text in df["text"]:
        text = str(text)
        text, n = re.subn(r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]", text)
        email_count += n
        text, n = re.subn(r"\b[6-9]\d{9}\b", "[PHONE]", text)
        phone_count += n
        text, n = re.subn(r"ORD-\d{6}", "[ORDER_ID]", text)
        order_count += n
        redacted_texts.append(text)
    results = []
    messages_list = []
    for i, text in enumerate(redacted_texts):
        original = str(df["text"][i])
        if original.strip() == "":
            status = "skipped_empty"
            messages_list.append(None)
        elif est_tokens(original) > MAX_INPUT_TOKENS:
            status = "flagged_long"
            messages_list.append(None)
        else:
            status = "pending"
            messages_list.append([{"role": "user", "content": text}])
        results.append({
            "ticket_id": df["ticket_id"][i],
            "status": status,
            "est_tokens": est_tokens(original) if original.strip() else 0,
        })
    for j, m in enumerate(messages_list):
        if m is not None:
            assert m[0]["content"] == redacted_texts[j]
            assert len(m) == 1
    out = pd.DataFrame(results)
    print("----------------------- Input Guardrail ---------------------------")
    print(out["status"].value_counts())
    print(out[out["status"] == "flagged_long"]["est_tokens"].tolist())
    print("\n----------------------- Redaction list --------------------------")
    print("Number of Substitutions for EMAILS :", email_count)
    print("Number of Substitutions for PHONES :", phone_count)
    print("Number of Substitutions for ORDER IDs :", order_count)
    print("Total number of substitutions :", email_count + phone_count + order_count)
    left = 0
    for text in redacted_texts:
        left += len(re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text))
        left += len(re.findall(r"\b[6-9]\d{9}\b", text))
        left += len(re.findall(r"ORD-\d{6}", text))
    print("left after redaction:", left)
    return df, messages_list            # <-- hand these to the other functions
# ---------------- Task 3: prompt checks ----------------
def prompts():
    for p in (ZERO_SHOT_PROMPT, FEW_SHOT_PROMPT):
        assert "CLASSIFY" in p
        for c in CATEGORIES:
            assert c in p
        assert '{"category": "...", "reason": "..."}' in p
    assert "EXAMPLES:" not in ZERO_SHOT_PROMPT
    assert "EXAMPLES:" in FEW_SHOT_PROMPT
    assert FEW_SHOT_PROMPT.count("Answer:") >= 2
    assert '"category": "fraud"' in FEW_SHOT_PROMPT
    print("\n------------------- Prompt Creation -----------------------")
    print("Both prompts satisfy the conditions.")
# ---------------- Task 4: retry function (top level, so budget() can use it too) ----------------
def call_with_retry(messages, ticket_id, log):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            log["errors"] += 1
            if attempt == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2**attempt * random.uniform(0.9, 1.15)
            log["lines"].append(f"[retry] ticket={ticket_id} retry={attempt} wait={wait:.4f}s")
            time.sleep(wait)
# ---------------- Task 5: validation function (top level) ----------------
def validate(text):
    status, category = "flagged_invalid_output", None
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("category") in CATEGORIES:
            status, category = "classified_json", data["category"]
    except json.JSONDecodeError:
        found = [w for w in re.findall(r"[a-z]+", text.lower()) if w in CATEGORIES]
        if len(found) == 1:
            status, category = "classified_keyword", found[0]
    return status, category
def run_pipeline(system_prompt, df, messages_list):
    mock_llm.reset()
    log = {"errors": 0, "lines": []}
    permanent = 0
    rows, total_tokens = [], 0
    for i, m in enumerate(messages_list):
        if m is None:
            continue
        tid = df["ticket_id"][i]
        messages = [{"role": "system", "content": system_prompt}] + m
        try:
            response = call_with_retry(messages, tid, log)
        except TransientLLMError:
            permanent += 1
            continue
        total_tokens += response.usage.total_tokens
        status, category = validate(response.choices[0].message.content or "")
        rows.append({"ticket_id": tid, "true": df["true_category"][i],
                     "pred": category, "status": status})
    return pd.DataFrame(rows), total_tokens, log, permanent
def retry_validate(df, messages_list):
    zero_res, zero_tokens, zero_log, zero_perm = run_pipeline(ZERO_SHOT_PROMPT, df, messages_list)
    few_res, few_tokens, few_log, few_perm = run_pipeline(FEW_SHOT_PROMPT, df, messages_list)
    print("\n-------------------- Retry Mechanism ----------------------")
    for line in zero_log["lines"]:
        print(line)
    print("tickets answered  :", len(zero_res))
    print("permanent errors  :", zero_perm)
    print("TransientLLMErrors:", zero_log["errors"])
    print("retry log lines   :", len(zero_log["lines"]))
    print("\n-------------------- Output Validation ----------------------")
    print(zero_res["status"].value_counts())
    print("Ticket IDs routed for human review:",
          zero_res[zero_res["status"] == "flagged_invalid_output"]["ticket_id"].tolist())
    print("\n-------------- Prompts Comparison with Confusion Table --------------------")
    for name, res, tok in [("Zero-shot", zero_res, zero_tokens), ("Few-shot", few_res, few_tokens)]:
        cls = res[res["status"].isin(["classified_json", "classified_keyword"])]
        n = len(cls)
        ok = int((cls["true"] == cls["pred"]).sum())
        print(f"{name}: classified={n}, accuracy={ok/n*100:.1f}% ({ok} of {n}), "
              f"total tokens={tok}, tokens per classified={tok/n:.1f}")
    zero_cls = zero_res[zero_res["status"].isin(["classified_json", "classified_keyword"])]
    print("\nZero-shot confusion table (rows = true, columns = predicted):")
    print(pd.crosstab(zero_cls["true"], zero_cls["pred"]))
# ---------------- Task 7: budget, routing, output ----------------
def budget(df, messages_list):
    os.makedirs("outputs", exist_ok=True)
    seg = pd.read_csv("outputs/customer_segments.csv", keep_default_na=False, dtype=str)
    seg = seg.set_index("customer_id")
    def route_ticket(status, category, is_premium):
        if status in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
            return "human_review", "NORMAL"
        if category == "fraud":
            return "security_team", "HIGH"
        if is_premium:
            return "premium_desk", "HIGH"
        return "support_queue", "NORMAL"
    def run_with_budget(system_prompt, token_budget):
        mock_llm.reset()
        log = {"errors": 0, "lines": []}
        running_total = 0
        rows = []
        for i, m in enumerate(messages_list):
            tid = df["ticket_id"][i]
            cid = str(df["customer_id"][i])
            category, tokens_used = "", 0
            if m is None:
                status = "skipped_empty" if str(df["text"][i]).strip() == "" else "flagged_long"
            elif running_total >= token_budget:
                status = "budget_exhausted"
            else:
                messages = [{"role": "system", "content": system_prompt}] + m
                try:
                    response = call_with_retry(messages, tid, log)
                    tokens_used = response.usage.total_tokens
                    running_total += tokens_used
                    status, cat = validate(response.choices[0].message.content or "")
                    category = cat or ""
                except TransientLLMError:
                    status = "flagged_invalid_output"
            if cid in seg.index:
                segment_name = seg.loc[cid, "segment_name"]
                is_premium = str(seg.loc[cid, "is_premium"]).strip().lower() in ("true", "1", "yes")
            else:
                segment_name, is_premium = "", False
            route, priority = route_ticket(status, category, is_premium)
            rows.append({"ticket_id": tid, "customer_id": cid, "status": status,
                         "category": category, "segment_name": segment_name,
                         "priority": priority, "route": route, "tokens_used": tokens_used})
        return pd.DataFrame(rows)
    # (a) budget demo: zero-shot
    TOKEN_BUDGET = 6000
    demo = run_with_budget(ZERO_SHOT_PROMPT, TOKEN_BUDGET)
    demo.to_csv("outputs/triage_budget_demo.csv", index=False)
    print("\n-------------------- Budget Demo (zero-shot) ----------------------")
    print("budget used            :", TOKEN_BUDGET)
    print("budget_exhausted count :", int((demo["status"] == "budget_exhausted").sum()))
    print("tokens used in demo    :", demo["tokens_used"].sum())
    # (b) final output: few-shot
    final = run_with_budget(FEW_SHOT_PROMPT, 100000)
    final.to_csv("outputs/triage_results.csv", index=False)
    print("\n-------------------- Final Results (few-shot) ----------------------")
    print("rows saved:", len(final))
    print("\nStatus counts:")
    print(final["status"].value_counts())
    print("\nRoute counts:")
    print(final["route"].value_counts())
    print("\nPriority counts:")
    print(final["priority"].value_counts())

if __name__ == "__main__":
    df, messages_list = guardrail_redact()
    prompts()
    retry_validate(df, messages_list)
    budget(df, messages_list)