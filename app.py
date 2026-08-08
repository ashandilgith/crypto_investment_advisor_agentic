import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from src.state import CryptoState
from src.agents.specialists import macro_orchestrator_node
from src.agents.analyst import analyst_node
from src.agents.trainer import test_evaluator_node, trainer_node
from src.main import build_workflow, load_persistent_weights
from src.pdf_generator import create_pdf_report

# Streamlit Page Setup
st.set_page_config(
    page_title="Verdict | Multi-Agentic Crypto Expert",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Minimalist CSS Styling
st.markdown("""
<style>
    .main { background-color: #FAFAFA; }
    .stButton>button {
        width: 100%;
        background-color: #1A365D;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.6rem 1rem;
    }
    .stButton>button:hover { background-color: #2B6CB0; color: white; }
    .metric-card {
        background-color: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# AUTHENTICATION GUARD
# ---------------------------------------------------------
EXPECTED_PASSWORD = os.getenv("APP_PASSWORD", "admin123")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Locusts")
    st.subheader("Multi-Agent Crypto Investment Intelligence System")
    
    pwd_input = st.text_input("Enter Access Key:", type="password")
    if st.button("Unlock Dashboard"):
        if pwd_input == EXPECTED_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid Access Key. Please check your .env configuration.")
    st.stop()

# ---------------------------------------------------------
# MAIN DASHBOARD INTERFACE
# ---------------------------------------------------------

st.title("Verdict - Multi-Agent Crypto Prediction")
st.caption("Powered by LangGraph, GPT-5.5 Reasoning, & Multi-Agent Parallel Macro Networks")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Execution Panel")
    st.markdown("---")
    
    selected_coins = st.multiselect(
        "GCash Assets to Scan:",
        ["BTC", "ETH", "SOL", "AVAX", "DOT", "LINK"],
        default=["BTC", "ETH", "SOL", "AVAX", "DOT", "LINK"]
    )
    
    st.markdown("---")
    st.markdown("**System Architecture:**")
    st.markdown("- **Pass 1:** $T-28$ Historical Backtest")
    st.markdown("- **Pass 2:** $T-14$ Historical Backtest")
    st.markdown("- **Pass 3:** Live $T+1$ Execution")
    
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

# Execution Trigger
if st.button("Run 3-Pass Agentic Analysis"):
    learning_dir = PROJECT_ROOT / "predictions_and_learning"
    initial_weights = load_persistent_weights(learning_dir)
    
    from datetime import datetime, timedelta
    t_minus_28 = (datetime.now() - timedelta(days=28)).strftime('%Y-%m-%d')
    t_minus_14 = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
    
    initial_state = {
        "current_evaluation_date": t_minus_28,
        "pending_evaluation_dates": [t_minus_14],
        "is_live_mode": False,
        "gcash_coins": selected_coins,
        "benchmark_stocks": ["SPY", "QQQ"],
        "macro_matrix": {},
        "current_weights": initial_weights,
        "prediction_history": [],
        "learned_adjustments": [],
        "final_recommendation": ""
    }
    
    app = build_workflow()
    
    with st.spinner("Fanning out specialist micro-agents, training weights, and invoking GPT-5.5..."):
        final_state = app.invoke(initial_state)
        
        # Translate to plain English using gpt-4o-mini
        mini_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        final_decision = json.loads(final_state["final_recommendation"])
        
        translation_prompt = f"""
        Summarize this technical recommendation in plain, simple English:
        Target Asset: {final_decision['predicted_best_coin']}
        Rationale: {final_decision['rationale']}
        """
        simple_translation = mini_llm.invoke([SystemMessage(content=translation_prompt)]).content
        
        # Save to Session State so the UI persists without re-running
        st.session_state.final_state = final_state
        st.session_state.final_decision = final_decision
        st.session_state.simple_translation = simple_translation
        
        # Generate PDF Bytes
        st.session_state.pdf_bytes = create_pdf_report(
            target_coin=final_decision['predicted_best_coin'],
            weights=final_state['current_weights'],
            gpt_55_output=final_decision['rationale'],
            gpt_4_translation=simple_translation,
            history=final_state.get("prediction_history", [])
        )

# ---------------------------------------------------------
# RESULTS DISPLAY (Renders immediately if available)
# ---------------------------------------------------------
if "final_decision" in st.session_state:
    decision = st.session_state.final_decision
    state = st.session_state.final_state
    
    st.markdown("---")
    
    # HIGHLIGHT CARD (Shows recommendation immediately before downloading PDF)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin:0; color:#718096;">RECOMMENDED ASSET TODAY</h4>
            <h1 style="margin:0; color:#1A365D; font-size: 3rem;">{decision['predicted_best_coin']}</h1>
            <p style="margin-top:8px; color:#2B6CB0; font-weight:600;">Status: High Conviction Allocation</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        # PDF DOWNLOAD BUTTON
        st.download_button(
            label="📄 Download PDF Market Report",
            data=st.session_state.pdf_bytes,
            file_name=f"Crypto_Recommendation_{decision['predicted_best_coin']}.pdf",
            mime="application/pdf"
        )

    with col2:
        st.subheader("⚖️ Applied Model Feature Weights")
        weights = state['current_weights']
        cols = st.columns(len(weights))
        for idx, (k, v) in enumerate(weights.items()):
            cols[idx].metric(label=k.replace('_', ' ').title(), value=f"{v*100:.1f}%")

    st.markdown("---")

    # DUAL-OUTPUT TABS
    tab1, tab2, tab3 = st.tabs(["🎯 Executive Summary (GPT-4 Translation)", "🔬 Technical Rationale (GPT-5.5 Engine)", "📊 Macro Matrix Data"])

    with tab1:
        st.markdown("### Executive Plain-English Summary")
        st.info(st.session_state.simple_translation)

    with tab2:
        st.markdown("### Deep GPT-5.5 Technical Rationale")
        st.write(decision['rationale'])

    with tab3:
        st.markdown("### Specialized Micro-Agent Matrix Scores")
        st.json(state['macro_matrix'])