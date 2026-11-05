from code_automation.data_audit import audit,data_clean,validate,cross_check,exploratory_analysis
from code_automation.RFM_modelbuilding import RFM_table,k_mean,lapsed_model
from code_automation.triage import guardrail_redact, prompts, retry_validate, budget
from code_automation.rag_agents import policy_rag, evaluate_retrieval, my_agent

def main():
    print("\n =========== PART A: AUDIT / CLEAN / VALIDATE =========")
    audit(); data_clean(); validate(); cross_check(); exploratory_analysis()
    print("\n =========== PART B: RFM TABLE, K -MEAN MODEL, LAPSE MODEL =========")
    rfm = RFM_table(); k_mean(rfm); lapsed_model()
    print("\n =========== PART C: GUARDRAIL AND TRIAGE PIPELINE =========")
    df, messages_list = guardrail_redact(); prompts()
    retry_validate(df, messages_list); budget(df, messages_list)
    print("\n============= PART D: POLICY RAG ===============")
    policy_rag(); evaluate_retrieval(); my_agent()

if __name__=="__main__": main()