import asyncio
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from src.state import CryptoState
from src.tools import fetch_historical_news, fetch_multi_timeframe_data

mini_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

async def _run_specialist(domain: str, query: str, target_date: str, data_type: str = "news", tickers: list = None) -> dict:
    """Executes a single micro-agent with dynamic scoring logic."""
    raw_data = ""
    if data_type == "market_data":
        raw_data = fetch_multi_timeframe_data.invoke({"tickers": tickers, "target_date_str": target_date})
    else:
        raw_data = fetch_historical_news.invoke({"query": query, "target_date_str": target_date})
    
    prompt = f"""
    You are an expert market analyst specializing in: {domain}.
    Target Date: {target_date}
    Raw Data: {raw_data}

    Evaluate the impact of this data on crypto liquidity on a scale from 1 to 10:
    - 1-3: Strongly Bearish / Hawkish / Economic Contraction
    - 4: Mildly Bearish
    - 5: Perfectly Neutral
    - 6: Mildly Bullish
    - 7-10: Strongly Bullish / Dovish / Expansion

    Output MUST be a valid JSON object matching this schema:
    {{
        "domain": "{domain}",
        "score": <INTEGER_BETWEEN_1_AND_10>,
        "summary": "<1_SENTENCE_OBJECTIVE_SUMMARY>"
    }}
    """
    
    try:
        response = await mini_llm.ainvoke([SystemMessage(content=prompt)])
        cleaned = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(cleaned)
        result["score"] = max(1, min(10, int(result.get("score", 5))))
        # Attach raw data so downstream nodes (GPT-5.5) can read the exact raw trajectory
        result["raw_data"] = raw_data 
        return result
        
    except Exception as e:
        return {
            "domain": domain, 
            "score": 5, 
            "summary": f"Fallback applied: {str(e)}",
            "raw_data": raw_data
        }

async def _gather_all(tasks):
    return await asyncio.gather(*tasks)

def macro_orchestrator_node(state: CryptoState) -> dict:
    date = state['current_evaluation_date']
    print(f"\n--- 🌍 ORCHESTRATOR Fanning out specialists for {date} ---")
    
    specialist_tasks = [
        ("US_Rates_Inflation", "US Federal reserve interest rates and CPI", "news", []),
        ("Europe_Rates_Inflation", "ECB interest rates and Europe inflation", "news", []),
        ("Japan_Rates_Inflation", "Bank of Japan interest rates and inflation", "news", []),
        ("Global_Energy", "Global crude oil WTI and natural gas prices", "news", []),
        ("US_Employment", "US non-farm payrolls and unemployment rate", "news", []),
        ("Crypto_Industry_News", "Major cryptocurrency news, regulatory sentiment, ETF flows", "news", []),
        ("Crypto_Historical_Trajectory", "", "market_data", state['gcash_coins']) 
    ]
    
    tasks = [_run_specialist(domain, query, date, dtype, tickers) for domain, query, dtype, tickers in specialist_tasks]
    results = asyncio.run(_gather_all(tasks))
    
    macro_matrix = {}
    for res in results:
        macro_matrix[res["domain"]] = {
            "score": res["score"],
            "summary": res["summary"],
            "raw_data": res.get("raw_data", "")
        }
        
    return {"macro_matrix": macro_matrix}