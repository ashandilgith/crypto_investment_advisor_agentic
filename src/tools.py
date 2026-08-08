import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from ddgs import DDGS
from langchain_core.tools import tool

@tool
def fetch_multi_timeframe_data(tickers: list[str], target_date_str: str) -> str:
    """Fetches 1d, 7d, 1m, 3m, 6m percentage returns for assets up to a specific date."""
    target_date = pd.to_datetime(target_date_str)
    yf_end_date = (target_date + timedelta(days=1)).strftime('%Y-%m-%d')
    
    report = [f"Market Trajectories up to {target_date_str}:"]
    
    for ticker in tickers:
        symbol = f"{ticker}-USD" if ticker not in ["SPY", "QQQ"] and not ticker.endswith("-USD") else ticker
        try:
            df = yf.Ticker(symbol).history(end=yf_end_date, period="7mo")
            if df.empty or len(df) < 130:
                report.append(f"{ticker}: Insufficient data.")
                continue
                
            closes = df['Close']
            latest_close = closes.iloc[-1]
            
            def get_return(days_back):
                if len(closes) > days_back:
                    past_close = closes.iloc[-(days_back + 1)]
                    return ((latest_close - past_close) / past_close) * 100
                return None

            r_1d = get_return(1)
            r_7d = get_return(5)   
            r_1m = get_return(21)  
            r_3m = get_return(63)  
            r_6m = get_return(126) 
            
            row = f"[{ticker}] 1d: {r_1d:.2f}% | 7d: {r_7d:.2f}% | 1m: {r_1m:.2f}% | 3m: {r_3m:.2f}% | 6m: {r_6m:.2f}% (Price: ${latest_close:.2f})"
            report.append(row)
        except Exception as e:
            report.append(f"{ticker}: Error fetching data - {str(e)}")
            
    return "\n".join(report)

@tool
def fetch_historical_news(query: str, target_date_str: str) -> str:
    """Fetches objective news context. Uses the date string to anchor the search context."""
    try:
        search_query = f"{query} {target_date_str}"
        results = DDGS().text(search_query, max_results=3)
        if not results:
            return "No news found."
        return "\n".join([f"- {r.get('title')}: {r.get('body')}" for r in results])
    except Exception as e:
        return f"News search failed: {str(e)}"