# Instamart Project
## Reproduce
1. Clone the repository and switch to the `main` branch:
git clone <repository-url>
cd Instamart_project
git checkout main

2. Create and activate a virtual environment:
python -m venv .venv
On Windows:
.venv\Scripts\activate
On macOS/Linux:
source .venv/bin/activate

3. Install the dependencies:
pip install -r requirements.txt
python generate_data.py
python mock_llm.py

## PART - A
1. Data Audit - Check for missing values, check the data type
2. Clean Dataset - immute the missing values, Change the datatype if required
3. Data Validate - using SQL quires, crossed the pandas tolal with SQL total using Pivot Table.
4. EDA Analysis - Analysis made through Histogram, Box-plot, Heatmap

## PART - B
1. RFM Table Creation which include features (recency, frequency, monetary)
2. K-Mean clustering in RFM table to segment the customers and found the best k as `K=4`. PCA - Feature Extraction is used for visualization
3. Lapse Model Building using lapsed dataset. Lapse features use orders strictly before 2025-09-01, while the label uses September outcomes
   Logistic Regression model
   Random Forest Model
`Chosen Random Forest Model` base on F1 score
4. Class - Balance experiment is made with Over Sampling Technique and observed that Recall rises but Precision do not rised. Feature Importance listed for both lapsed model and selected model is saved in .joblib
5. Limitations and Next Steps:
   The retention list is illustrative and not a leakage-free evaluation, because the Premium flag is a post-hoc whole-window label built from RFM that includes the September window that defines the lapse label ( the model’s own features still use only orders before the cutoff).

## PART - C
1. Tickets checked for Input guardrail and redaction is done with regular expressions and placeholders before it goes the prompt (Few-shot and Zero-shot)
2. Some call raises a error, and Retry methodology is used to fix temporary errors only.
3. Every response is validated with accepted category, and routed the tickets to respective departments.
4. Few-shot cost: it used `125.5` tokens on average per classified ticket, versus zero-shot cost `79.9` tokens used on average per ticket, an extra `45.6` tokens per ticket.
Rule-validated: 147 tickets, where 135 were classified_json (valid JSON, allowed category) and 12 were classified_keyword (exactly one category word found with re.findall). Sent to humans: 17 tickets in human_review, being 9 flagged_invalid_output (such as the shipping category), 4 skipped_empty and 4 flagged_long. `45.6` token cost more at the same time `accuracy is good` in Few-Shot.
In the budget demo (6000 tokens), the 76 budget_exhausted tickets also went to human review, but they are not part of the final results.

## PART - D
### Table 
 Scenario |         Status	         |  Refund (INR)  |   Model calls	 |  Tokens used
   S1	    |         Refund	         |     128.25	  |        5       |     1112 
   S2	    |         Refund          |     135.00	  |        5       |     1112 
   S3	    |        Escalate         |    1999.00	  |        5       |     1099 
   S4	    |         Denied	         |                |        4       |      777
   S5	    |         Denied          |                |        3       |      466
   S6	    |        Not_Found        |                |        2       |      158 
   S7	    | Fallback_Max_Iterations |	              |        6       |     3258 (about 2,700 to 5,500)

### Demos on loop termination:
D1 OUTPUT: Demo token_limit=200   -> FALLBACK_TOKEN_LIMIT | calls: 3 | steps: 2
           Handing off to a human agent. Reason: token limit exceeded. Steps completed: get_order, search_policy. Customer request: My order ORD-100048 reached me very late. I would like a refund.
D1 OBSERVATION: d1 (token_limit = 200; time_limit_seconds = default(10)) 
                After call 3 the token limit exceeds 200, so it stopped and did not run the tool from call 3. Only `get_order` and `search_policy` had run, so 2 steps and no refund --> Human agent.

D2 OUTPUT: Demo time_limit_seconds=0 -> FALLBACK_TIMEOUT | calls: 0 | steps: 0
           Handing off to a human agent. Reason: time limit reached. Steps completed: none. Customer request: My order ORD-100048 reached me very late. I would like a refund.
D2 OBSERVATION: d2 (token_limit = default(6000); time_limit_seconds = 0)
                With a limit of 0, elapsed >= 0 is true at the top of the first iteration. The agent stops before calling the model, so 0 calls and 0 steps.

### Demo on crash:
A crash is simulated after the 2nd tool execution and `resume_agent` is called to resume the process. Only four executions after the crash which give the final status.

get_order → search_policy → calculate_refund → issue_refund, with final status REFUND and 128.25.