import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from src.agents.specialists import macro_orchestrator_node
from src.agents.analyst import analyst_node
from src.agents.trainer import test_evaluator_node, trainer_node
from src.state import CryptoState

def route_analyst_output(state: CryptoState):
    if state["is_live_mode"]:
        return "END"
    return "TestEvaluator"

def build_workflow():
    workflow = StateGraph(CryptoState)
    workflow.add_node("Macro_Orchestrator", macro_orchestrator_node)
    workflow.add_node("Analyst", analyst_node)
    workflow.add_node("TestEvaluator", test_evaluator_node)
    workflow.add_node("Trainer", trainer_node)

    workflow.set_entry_point("Macro_Orchestrator")
    workflow.add_edge("Macro_Orchestrator", "Analyst")
    workflow.add_conditional_edges("Analyst", route_analyst_output, {"TestEvaluator": "TestEvaluator", "END": END})
    workflow.add_edge("TestEvaluator", "Trainer")
    workflow.add_edge("Trainer", "Macro_Orchestrator")

    return workflow.compile()

def load_persistent_weights(learning_dir: Path) -> dict:
    default_weights = {
        "western_macro": 0.15,
        "eastern_macro": 0.15,
        "global_indicators": 0.20,
        "crypto_historical_trajectory": 0.25,
        "crypto_sentiment": 0.25
    }
    
    if not learning_dir.exists():
        learning_dir.mkdir(parents=True, exist_ok=True)
        return default_weights

    saved_files = sorted(learning_dir.glob("test_*.json"))
    if saved_files:
        try:
            with open(saved_files[-1], "r", encoding="utf-8") as f:
                data = json.load(f)
                if "weights_used" in data and data["weights_used"]:
                    return data["weights_used"]
        except Exception:
            pass
            
    return default_weights

def generate_simple_log_report(final_state: dict, learning_dir: Path):
    print("\n--- 📝 GENERATING PLAIN-ENGLISH RUN LOG (gpt-4o-mini) ---")
    mini_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    final_decision = json.loads(final_state["final_recommendation"])
    history = final_state.get("prediction_history", [])
    
    history_str = ""
    for idx, test in enumerate(history):
        history_str += f"\nTEST {idx+1} (Date: {test.get('target_date', 'N/A')}):\n"
        history_str += f"- Predicted: {test.get('predicted_best_coin', 'N/A')} | Actual: {test.get('actual_best_coin', 'N/A')}\n"
        history_str += f"- Correct?: {test.get('was_correct', False)}\n"
        
    prompt = f"""
    You are a financial log generator. Summarize this 3-pass agent pipeline in simple English.
    
    1. HISTORICAL TESTS (PASS 1 & PASS 2):
       {history_str}
       Trainer's Final Lesson: {final_state.get('learned_adjustments', ['None'])[-1] if final_state.get('learned_adjustments') else 'None'}
       
    2. WEIGHT ADJUSTMENTS:
       Final Tuned Weights Applied for Live Decision: {json.dumps(final_state.get('current_weights', {}), indent=2)}
       
    3. LIVE DECISION FOR TODAY (PASS 3):
       Selected Asset: {final_decision.get('predicted_best_coin', 'N/A')}
       Rationale: {final_decision.get('rationale', '')}

    FORMAT REQUIREMENTS:
    ## 1. BACKTEST SUMMARY (HISTORICAL PASSES)
    Explain what happened during the 2 historical tests and what lessons were extracted.

    ## 2. WEIGHT ADJUSTMENTS
    Explain which data points the model is prioritizing today based on the tests.

    ## 3. WHAT TO BUY TODAY AND WHY
    Give a straightforward recommendation based on the Analyst's final output.
    """
    
    response = mini_llm.invoke([SystemMessage(content=prompt)]).content
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = learning_dir / f"run_log_{timestamp}.txt"
    
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(response)
        
    print(f"✅ Run log saved to: {log_filename}\n")

if __name__ == "__main__":
    app = build_workflow()
    
    learning_dir = PROJECT_ROOT / "predictions_and_learning"
    initial_weights = load_persistent_weights(learning_dir)
    
    # Calculate historical dates for the 3-pass system
    t_minus_28 = (datetime.now() - timedelta(days=28)).strftime('%Y-%m-%d')
    t_minus_14 = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
    
    print("Booting 3-Pass Adaptive Agent Pipeline...\n")
    
    initial_state = {
        "current_evaluation_date": t_minus_28,
        "pending_evaluation_dates": [t_minus_14], # Queues the second historical pass
        "is_live_mode": False,
        "gcash_coins": ["BTC", "ETH", "SOL", "AVAX", "DOT", "LINK"],
        "benchmark_stocks": ["SPY", "QQQ"],
        "macro_matrix": {},
        "current_weights": initial_weights,
        "prediction_history": [],
        "learned_adjustments": [],
        "final_recommendation": ""
    }
    
    final_state = app.invoke(initial_state)
    final_decision = json.loads(final_state["final_recommendation"])
    
    print("\n=========================================================")
    print("        FINAL LIVE CIO RECOMMENDATION (T+1)              ")
    print("=========================================================")
    print(f"Target Asset: {final_decision['predicted_best_coin']}")
    print(f"Tuned Weights Applied: {final_state['current_weights']}")
    print(f"Trainer's Lessons: {final_state['learned_adjustments'][-1] if final_state['learned_adjustments'] else 'None'}")
    print("\nRationale:")
    print(final_decision['rationale'])
    print("=========================================================")
    
    generate_simple_log_report(final_state, learning_dir)