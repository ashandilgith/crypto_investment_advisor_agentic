import operator
from typing import TypedDict, Annotated, List, Dict, Any

class PredictionRecord(TypedDict):
    target_date: str
    predicted_best_coin: str
    actual_best_coin: str
    was_correct: bool
    weights_used: Dict[str, float]
    rationale: str

class CryptoState(TypedDict):
    current_evaluation_date: str 
    pending_evaluation_dates: List[str]
    is_live_mode: bool
    
    gcash_coins: List[str]
    benchmark_stocks: List[str]
    
    # The Specialized Macro Intelligence Matrix
    macro_matrix: Dict[str, Any]
    
    # Dynamic Weights 
    current_weights: Dict[str, float]
    
    # Memory
    prediction_history: Annotated[List[PredictionRecord], operator.add]
    learned_adjustments: Annotated[List[str], operator.add]
    final_recommendation: str