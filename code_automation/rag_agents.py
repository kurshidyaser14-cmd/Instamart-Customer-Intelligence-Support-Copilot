import os
import pandas as pd
import mock_llm
import json
import sqlite3
import inspect
import time
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def policy_rag():
    global chunk_documents, retrieve, answer, chunks, vectorizer, chunk_matrix
    def chunk_documents(folder="data/policies", size=40, overlap=10):
        step = size - overlap
        chunks = []
        for fname in sorted(os.listdir(folder)):
            with open(os.path.join(folder, fname), encoding="utf-8") as f:
                words = f.read().split()
            start = 0
            chunk_id = 0
            while True:
                end = start + size
                chunks.append({
                    "source": fname,
                    "chunk_id": chunk_id,
                    "text": " ".join(words[start:end]),
                })
                if end >= len(words):
                    break
                start += step
                chunk_id += 1
        return chunks
    if True:
        print("----------------- Chunking ------------------")
        chunks = chunk_documents()
        counts = {}
        for c in chunks:
            counts[c["source"]] = counts.get(c["source"], 0) + 1
        for name, n in counts.items():
            print(name, n)
        print("Total chunks:", len(chunks))
    chunks = chunk_documents()
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer()
    chunk_matrix = vectorizer.fit_transform(texts)
    def retrieve(query = "refund policy late delivery", k=3):
        query_vec = vectorizer.transform([query])
        scores = cosine_similarity(query_vec, chunk_matrix)[0]
        top_idx = scores.argsort()[::-1][:k]
        return [{
                "source": chunks[i]["source"],
                "score": float(scores[i]),
                "text": chunks[i]["text"],}
            for i in top_idx]
    if True:
        print("----------------- Retrival Results -----------------")
        for r in retrieve():
            print(f" Source: {r['source']},\n Score: {round(r['score'], 3)}, \n Text: {r['text'][:100]}...\n")
    SYSTEM_PROMPT = (
        "You are a POLICY ASSISTANT. Answer ONLY from the supplied context. "
        "If the context does not contain the answer, say you could not find "
        "the information."
    )
    def answer(question):
        results = retrieve(question, k=3)
        context = "\n\n".join(f"[{r['source']}]\n{r['text']}" for r in results)
        user_msg = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
        reply = mock_llm.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ])
        text = reply.choices[0].message.content
        sources = []
        for r in results:
            if r["source"] not in sources:
                sources.append(r["source"])
        return text, sources
    if True:
        print("------------------ Auguement and Answer ------------------")
        q = "What is the refund policy for late delivery?"
        text, srcs = answer(q)
        print("Question:", q)
        print("Answer:", text)
        print("Sources:", srcs)

def evaluate_retrieval():
    def rebuild_index(overlap):
        # Re-create the Task 2 globals for a different overlap.
        global chunks, vectorizer, chunk_matrix
        chunks = chunk_documents(overlap=overlap)
        vectorizer = TfidfVectorizer()
        chunk_matrix = vectorizer.fit_transform([c["text"] for c in chunks])
    def evaluate(overlap, df):
        df = pd.read_csv("data/raw/policy_eval.csv", dtype=str, keep_default_na=False)
        rebuild_index(overlap)
        doc_hits = fact_hits = fallback_hits = 0
        n_ans = n_unans = 0
        failures = []
        for _, row in df.iterrows():
            text, sources = answer(row["question"])
            if row["expected_doc"].strip().upper() == "NONE":
                n_unans += 1
                if text.strip().startswith("I could not find"):
                    fallback_hits += 1
            else:
                n_ans += 1
                doc_ok = row["expected_doc"].strip() in sources
                fact_ok = row["expected_fact"] in text
                doc_hits += doc_ok
                fact_hits += fact_ok
                if not (doc_ok and fact_ok):
                    failures.append((row["question"], doc_ok, fact_ok))
        print(f"\n-------------- chunk size 40, overlap {overlap} ------------------")
        print(f"{'Check':<46}{'Result'}")
        print(f"{'expected_doc in top 3 (answerable)':<46}{doc_hits}/{n_ans}")
        print(f"{'answer contains expected_fact (answerable)':<46}{fact_hits}/{n_ans}")
        print(f"{'fallback \"I could not find\" (unanswerable)':<46}{fallback_hits}/{n_unans}")
        if failures:
            print("\nFailing questions:")
            for q, d, f in failures:
                print(f"- {q}  (doc found: {d}, fact found: {f})")
    if True:
        df = pd.read_csv("data/raw/policy_eval.csv", dtype=str, keep_default_na=False)
        evaluate(overlap=10, df=df)
        evaluate(overlap=0, df=df)
        rebuild_index(10)

def my_agent():
    DB_PATH = "data/instamart.db"
    LEDGER_PATH = "outputs/refund_ledger.jsonl"
    def get_order(order_id):
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        db_row = conn.execute(
            "SELECT order_id, status, order_amount, delivery_minutes "
            "FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        conn.close()
        if db_row is None:
            return {"error": "order not found"}
        return {
            "order_id": db_row["order_id"],
            "status": db_row["status"],
            "order_amount": db_row["order_amount"],
            "delivery_minutes": db_row["delivery_minutes"], }
    def search_policy(query):
        return {"results": retrieve(query)}
    def calculate_refund(order_amount, delivery_minutes):
        if delivery_minutes > 45:
            return {"refund_inr": order_amount}
        else:
            return {"refund_inr": 0}
    def issue_refund(order_id, refund_inr):
        return _write_ledger("issue_refund", order_id, refund_inr)

    def escalate_to_human(order_id, refund_inr):
        return _write_ledger("escalate_to_human", order_id, refund_inr)
    def _write_ledger(tool_name, order_id, refund_inr):
        os.makedirs("outputs", exist_ok=True)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "order_id": order_id,
            "refund_inr": refund_inr,
        }
        with open(LEDGER_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return {"refund_inr": refund_inr}
    TOOL_MAP = {
        "get_order": get_order,
        "search_policy": search_policy,
        "calculate_refund": calculate_refund,
        "issue_refund": issue_refund,
        "escalate_to_human": escalate_to_human,
    }
    def _spec(name, description, properties):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties.keys()),
                },
            },
        }
    TOOLS_SPEC = [
        _spec("get_order",
            "Look up an order by its ID and return its status, amount and delivery time.",
            {"order_id": {"type": "integer", "description": "The order ID"}}),
        _spec("search_policy",
            "Search the policy documents and return the most relevant text chunks.",
            {"query": {"type": "string", "description": "Search query"}}),
        _spec("calculate_refund",
            "Compute the refund in INR from the order amount and delivery minutes.",
            {"order_amount": {"type": "number", "description": "Order amount in INR"},
            "delivery_minutes": {"type": "number", "description": "Delivery time in minutes"}}),
        _spec("issue_refund",
            "Issue an automatic refund for an order and record it in the ledger.",
            {"order_id": {"type": "integer", "description": "The order ID"},
            "refund_inr": {"type": "number", "description": "Refund amount in INR"}}),
        _spec("escalate_to_human",
            "Escalate a refund above INR 1000 to a human and record it in the ledger.",
            {"order_id": {"type": "integer", "description": "The order ID"},
            "refund_inr": {"type": "number", "description": "Refund amount in INR"}}),
    ]
    def call_tool(name, arguments):
        func = TOOL_MAP[name]
        return func(**arguments)
    if True:
        print("\n------------------Tools defined -------------------")
        print("Tools are defined with the tool map is Successfully created.")
    SYSTEM_PROMPT = (
        "You are a refund AGENT for Instamart. Look up the order, search the policy, "
        "calculate the refund, then issue it or escalate it. "
        "Only completed orders can be refunded. Refunds above INR 1000 must be escalated. "
        "Finish with REFUND:, ESCALATE:, DENIED: or NOT_FOUND:."
    )
    class SimulatedCrash(Exception):
        pass
    def save_checkpoint(state):
        os.makedirs("state", exist_ok=True)
        with open(f"state/{state['run_id']}.json", "w", encoding="utf-8") as f:
            json.dump(state, f)
    def load_checkpoint(run_id):
        with open(f"state/{run_id}.json", encoding="utf-8") as f:
            return json.load(f)
    def agent_loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools):
        start = time.monotonic()
        def fallback(status, reason):
            done = ", ".join(state["tools_executed"]) or "none"
            summary = (f"Handing off to a human agent. Reason: {reason}. "
                    f"Steps completed: {done}. Customer request: {state['user_message']}")
            return {"status": status, "final_answer": summary, "steps": state["steps"],
                    "llm_calls": state["llm_calls"], "tokens_used": state["tokens_used"]}
        # continue from where the saved model-call count left off
        while state["llm_calls"] < max_iterations:
            if time.monotonic() - start >= time_limit_seconds:
                return fallback("FALLBACK_TIMEOUT", "time limit reached")
            reply = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
            state["llm_calls"] += 1
            state["tokens_used"] += reply.usage.total_tokens
            msg = reply.choices[0].message
            if state["tokens_used"] > token_limit:
                return fallback("FALLBACK_TOKEN_LIMIT", "token limit exceeded")
            if not msg.tool_calls:
                final = msg.content
                return {"status": final.split(":")[0].strip(), "final_answer": final,
                        "steps": state["steps"], "llm_calls": state["llm_calls"],
                        "tokens_used": state["tokens_used"]}
            calls = []
            for tc in msg.tool_calls:
                args = tc.arguments
                if isinstance(args, str):
                    args = json.loads(args)
                calls.append({"id": tc.id, "name": tc.name, "arguments": args})
            state["messages"].append({"role": "assistant", "content": None, "tool_calls": calls})
            for c in calls:
                result = call_tool(c["name"], c["arguments"])
                state["steps"].append({"tool": c["name"], "arguments": c["arguments"],
                                    "observation": result})
                state["messages"].append({"role": "tool", "tool_call_id": c["id"],
                                        "name": c["name"], "content": json.dumps(result)})
                state["tools_executed"].append(c["name"])
                state["step_number"] = len(state["tools_executed"])
                save_checkpoint(state) 
                if crash_after_tools == len(state["tools_executed"]):
                    raise SimulatedCrash(f"crash after tool {c['name']}")
        return fallback("FALLBACK_MAX_ITERATIONS", "max iterations reached")
    def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
                time_limit_seconds=10, run_id=None, crash_after_tools=None):
        state = {
            "run_id": run_id or f"{scenario_id}_{int(time.time())}",
            "scenario_id": scenario_id,
            "user_message": user_message,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message}],
            "tokens_used": 0,
            "llm_calls": 0,
            "step_number": 0,
            "tools_executed": [],
            "steps": [],
        }
        return agent_loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools)
    # ---------- continue a saved run ----------
    def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
        state = load_checkpoint(run_id)
        return agent_loop(state, max_iterations, token_limit, time_limit_seconds, None)
    if True:
        os.makedirs("outputs", exist_ok=True)
        open("outputs/refund_ledger.jsonl", "w").close()
        open("outputs/agent_runs.jsonl", "w").close()
        df = pd.read_csv("data/raw/agent_scenarios.csv", dtype=str, keep_default_na=False)
        print("\n---------- All 7 Scenarios Table -------\n")
        print("| Scenario | Status | Refund (INR) | Model calls | Tokens |")
        print("|---|---|---|---|---|")
        for _, row in df.iterrows():
            run = run_agent(row["message"], row["scenario_id"])
            amount = ""
            for s in run["steps"]:
                if s["tool"] in ("issue_refund", "escalate_to_human"):
                    amount = f"{float(s['observation']['refund_inr']):.2f}"
            with open("outputs/agent_runs.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "scenario_id": row["scenario_id"],
                    "status": run["status"],
                    "final_answer": run["final_answer"],
                    "llm_calls": run["llm_calls"],
                    "tokens_used": run["tokens_used"],
                    "steps": run["steps"],
                }) + "\n")
            print(f"| {row['scenario_id']} | {run['status']} | {amount} | "
                f"{run['llm_calls']} | {run['tokens_used']} |")
        print("\n---------- Limit Demos ----------")
        s1 = df[df["scenario_id"] == "S1"].iloc[0]["message"]
        d1 = run_agent(s1, "S1", token_limit=200)
        print("\nDemo token_limit=200   ->", d1["status"],
            "| calls:", d1["llm_calls"], "| steps:", len(d1["steps"]))
        print(d1["final_answer"])
        d2 = run_agent(s1, "S1", time_limit_seconds=0)
        print("\nDemo time_limit_seconds=0 ->", d2["status"],
            "| calls:", d2["llm_calls"], "| steps:", len(d2["steps"]))
        print(d2["final_answer"])
        print("\n---------- Crash and Resume ----------")
        try:
            run_agent(s1, "S1", run_id="S1_demo", crash_after_tools=2)
        except SimulatedCrash as e:
            print("\nCrashed:", e)
        result = resume_agent("S1_demo")
        refund = ""
        for s in result["steps"]:
            if s["tool"] == "issue_refund":
                refund = f"{float(s['observation']['refund_inr']):.2f}"
        print("Status:", result["status"], "| Refund:", refund)
        print("Tools executed (in order):", [s["tool"] for s in result["steps"]])

if __name__=="__main__": policy_rag(); evaluate_retrieval(); my_agent()