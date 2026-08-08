import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from src.state import CryptoState

expert_llm = ChatOpenAI(model="gpt-5.5", temperature=0.1)

def analyst_node(state: CryptoState) -> dict:
    print(f"--- 🧠 ANALYST EVALUATING ({state['current_evaluation_date']}) ---")
    
    macro_matrix = state['macro_matrix']
    
    # Extract the raw multi-timeframe trajectory table directly for GPT-5.5
    raw_trajectory_table = macro_matrix.get("Crypto_Historical_Trajectory", {}).get("raw_data", "Trajectory unavailable.")
    
    prompt = f"""
    Target Date for Prediction: {state['current_evaluation_date']} (Predicting for T+1)
    
    === 1. FULL ASSET TRAJECTORY SNAPSHOT (1d, 7d, 1m, 3m, 6m) ===
    {raw_trajectory_table}
    
    === 2. GLOBAL MACRO & REGIONAL SCORES ===
    {json.dumps(macro_matrix, indent=2)}
    
    === 3. APPLIED FEATURE WEIGHTS ===
    {json.dumps(state['current_weights'], indent=2)}
    
    === 4. LEARNED ADJUSTMENTS FROM PREVIOUS TESTS ===
    {chr(10).join(state['learned_adjustments']) if state['learned_adjustments'] else 'None'}
    
    Task: Examine the trajectory of EVERY coin alongside the macro conditions and applied weights. 
    Select exactly ONE coin from {state['gcash_coins']} to buy for T+1.
    
    Output pure JSON format:
    {{
        "predicted_best_coin": "TICKER",
        "rationale": "Detailed critical rationale explicitly referencing the multi-timeframe trajectories (1d, 7d, 1m, 3m, 6m) of the coins alongside macro weights."
    }}
    """
    
    response = expert_llm.invoke([SystemMessage(content=prompt)]).content
    cleaned = response.replace("```json", "").replace("```", "").strip()
    return {"final_recommendation": json.dumps(json.loads(cleaned))}