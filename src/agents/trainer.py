import os
import json
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from src.state import CryptoState, PredictionRecord

mini_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

def test_evaluator_node(state: CryptoState) -> dict:
    print("--- ⚖️ EVALUATOR CHECKING ACCURACY ---")
    
    prediction = json.loads(state['final_recommendation'])
    target_date = pd.to_datetime(state['current_evaluation_date'])
    next_day = (target_date + timedelta(days=1)).strftime('%Y-%m-%d')
    next_day_plus_one = (target_date + timedelta(days=2)).strftime('%Y-%m-%d')
    
    actual_returns = {}
    for coin in state['gcash_coins']:
        try:
            df = yf.Ticker(f"{coin}-USD").history(start=next_day, end=next_day_plus_one)
            if not df.empty and len(df) >= 1:
                ret = ((df['Close'].iloc[0] - df['Open'].iloc[0]) / df['Open'].iloc[0]) * 100
                actual_returns[coin] = ret
        except:
            continue
            
    actual_best_coin = max(actual_returns, key=actual_returns.get) if actual_returns else "UNKNOWN"
    was_correct = prediction["predicted_best_coin"] == actual_best_coin
    
    record = PredictionRecord(
        target_date=state['current_evaluation_date'],
        predicted_best_coin=prediction["predicted_best_coin"],
        actual_best_coin=actual_best_coin,
        was_correct=was_correct,
        weights_used=state['current_weights'],
        rationale=prediction["rationale"]
    )
    
    os.makedirs("predictions_and_learning", exist_ok=True)
    with open(f"predictions_and_learning/test_{state['current_evaluation_date']}.json", "w") as f:
        json.dump(record, f, indent=4)
        
    return {"prediction_history": [record]}

def trainer_node(state: CryptoState) -> dict:
    print("--- 🏋️ TRAINER REBALANCING MACRO & CRYPTO WEIGHTS ---")
    last_test = state['prediction_history'][-1]
    
    prompt = f"""
    Analyst was Correct?: {last_test['was_correct']}
    Predicted: {last_test['predicted_best_coin']} | Actual: {last_test['actual_best_coin']}
    
    Previous Weights: {json.dumps(last_test['weights_used'])}
    
    Adjust the weights across these 5 master categories (must sum exactly to 1.0). 
    Increase the weight for categories that correctly signaled the actual best coin, and decrease weights for noisy categories.
    
    Categories:
    - "western_macro" (US/Europe rates, inflation)
    - "eastern_macro" (Japan, SE Asia, Middle East)
    - "global_indicators" (Energy, tech, health, employment)
    - "crypto_historical_trajectory" (1d, 7d, 1m, 3m, 6m price action)
    - "crypto_sentiment" (Protocol upgrades, crypto regulation, industry news)
    
    Output pure JSON:
    {{
        "new_weights": {{"western_macro": float, "eastern_macro": float, "global_indicators": float, "crypto_historical_trajectory": float, "crypto_sentiment": float}},
        "learned_rule": "String"
    }}
    """
    
    response = mini_llm.invoke([SystemMessage(content=prompt)]).content
    result = json.loads(response.replace("```json", "").replace("```", "").strip())
    
    pending = state.get("pending_evaluation_dates", [])
    if pending:
        return {
            "current_weights": result["new_weights"],
            "learned_adjustments": [result["learned_rule"]],
            "current_evaluation_date": pending[0],
            "pending_evaluation_dates": pending[1:],
            "is_live_mode": False
        }
    else:
        return {
            "current_weights": result["new_weights"],
            "learned_adjustments": [result["learned_rule"]],
            "current_evaluation_date": datetime.now().strftime('%Y-%m-%d'),
            "is_live_mode": True
        }