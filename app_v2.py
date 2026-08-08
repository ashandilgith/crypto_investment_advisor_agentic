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

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(
    page_title="VERDICT | AI Investment Decision Engine",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# VERDICT / CRYPTO FINTECH THEME
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        /* ---------- Global ---------- */
        :root {
            --bg: #05070b;
            --panel: #0b1018;
            --panel-2: #0e1520;
            --border: #253243;
            --gold: #f5b82e;
            --gold-soft: #ffd86a;
            --green: #39ff88;
            --cyan: #20d9ff;
            --purple: #b66cff;
            --orange: #ff9f1c;
            --text: #f4f7fb;
            --muted: #8f9bad;
        }

        .stApp {
            background:
                radial-gradient(circle at 78% 5%, rgba(32, 217, 255, 0.07), transparent 25%),
                radial-gradient(circle at 12% 10%, rgba(245, 184, 46, 0.07), transparent 23%),
                linear-gradient(180deg, #04060a 0%, #070b11 52%, #04060a 100%);
            color: var(--text);
        }

        .main .block-container {
            max-width: 1500px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        [data-testid="stHeader"] {
            background: rgba(4, 6, 10, 0.85);
        }

        [data-testid="stSidebar"] {
            background: #070a10;
            border-right: 1px solid #202b39;
        }

        [data-testid="stSidebar"] * {
            color: var(--text);
        }

        /* ---------- Typography ---------- */
        h1, h2, h3, h4, p, label, .stMarkdown {
            color: var(--text);
        }

        .verdict-brand {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 4px;
        }

        .verdict-mark {
            width: 42px;
            height: 42px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border: 2px solid var(--gold);
            color: var(--gold);
            font-size: 25px;
            font-weight: 900;
            transform: rotate(45deg);
            box-shadow: 0 0 18px rgba(245,184,46,.18);
        }

        .verdict-mark span {
            transform: rotate(-45deg);
        }

        .verdict-word {
            font-size: 2.25rem;
            line-height: 1;
            letter-spacing: .22em;
            font-weight: 800;
            color: var(--gold);
            text-shadow: 0 0 20px rgba(245,184,46,.18);
        }

        .verdict-subtitle {
            color: #d8dee8;
            font-size: .78rem;
            letter-spacing: .24em;
            text-transform: uppercase;
            margin: 7px 0 0 58px;
        }

        .hero-copy {
            margin-top: 28px;
            margin-bottom: 26px;
        }

        .hero-copy .eyebrow {
            color: var(--gold);
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .18em;
            text-transform: uppercase;
        }

        .hero-copy h1 {
            font-size: clamp(2rem, 4vw, 3.8rem);
            line-height: 1.02;
            margin: 8px 0;
            letter-spacing: -.03em;
        }

        .hero-copy .accent {
            color: var(--gold);
        }

        .hero-copy p {
            color: var(--muted);
            font-size: 1rem;
            max-width: 760px;
            margin: 0;
        }

        /* ---------- Cards ---------- */
        .metric-card, .panel-card, .feature-card {
            background: linear-gradient(145deg, rgba(14,21,32,.98), rgba(7,11,17,.98));
            border: 1px solid var(--border);
            border-radius: 14px;
            box-shadow: 0 12px 40px rgba(0,0,0,.24), inset 0 1px 0 rgba(255,255,255,.025);
        }

        .metric-card {
            padding: 1.45rem 1.55rem;
            min-height: 205px;
        }

        .panel-card {
            padding: 1.25rem 1.4rem;
        }

        .feature-card {
            padding: 1.1rem;
            min-height: 150px;
        }

        .section-kicker {
            color: var(--gold);
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .16em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .asset-symbol {
            font-size: 3.3rem;
            line-height: 1;
            font-weight: 900;
            letter-spacing: -.04em;
            color: var(--gold);
        }

        .status-pill {
            display: inline-block;
            margin-top: 12px;
            padding: 6px 12px;
            border-radius: 999px;
            border: 1px solid rgba(57,255,136,.38);
            background: rgba(57,255,136,.07);
            color: var(--green);
            font-size: .75rem;
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;
        }

        .decision-banner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
            padding: 18px 22px;
            border: 1px solid rgba(245,184,46,.35);
            border-radius: 12px;
            background: linear-gradient(90deg, rgba(245,184,46,.08), rgba(245,184,46,.02));
            margin-bottom: 16px;
        }

        .decision-title {
            font-size: 2.2rem;
            font-weight: 900;
            color: var(--green);
            letter-spacing: .03em;
        }

        .feature-title {
            font-size: .9rem;
            font-weight: 800;
            letter-spacing: .04em;
            margin-bottom: 8px;
        }

        .feature-value {
            font-size: 1.65rem;
            font-weight: 800;
            color: var(--gold);
        }

        .feature-copy {
            color: var(--muted);
            font-size: .8rem;
            line-height: 1.45;
        }

        .weight-card {
            background: #090e16;
            border: 1px solid #202c3a;
            border-radius: 10px;
            padding: 12px;
            text-align: center;
        }

        .weight-label {
            color: var(--muted);
            font-size: .68rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            min-height: 32px;
        }

        .weight-value {
            color: var(--gold);
            font-size: 1.25rem;
            font-weight: 800;
        }

        .market-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }

        .market-chip {
            border: 1px solid #263546;
            background: #0b111a;
            color: #c7d0dc;
            border-radius: 999px;
            padding: 5px 10px;
            font-size: .7rem;
        }

        /* ---------- Streamlit controls ---------- */
        .stButton > button, .stDownloadButton > button {
            width: 100%;
            min-height: 44px;
            border-radius: 9px;
            border: 1px solid rgba(245,184,46,.55);
            background: linear-gradient(135deg, #d89b19, #f5b82e 52%, #ffd86a);
            color: #05070b;
            font-weight: 850;
            letter-spacing: .02em;
            box-shadow: 0 5px 18px rgba(245,184,46,.12);
        }

        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: #ffe08b;
            background: linear-gradient(135deg, #f5b82e, #ffe08b);
            color: #05070b;
            box-shadow: 0 0 22px rgba(245,184,46,.22);
        }

        .stTextInput input, .stMultiSelect div[data-baseweb="select"] {
            background: #0b1119 !important;
            color: var(--text) !important;
            border-color: #273545 !important;
        }

        .stMultiSelect span, .stSelectbox span {
            color: var(--text) !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid #243142;
        }

        .stTabs [data-baseweb="tab"] {
            color: #8f9bad;
            font-weight: 700;
        }

        .stTabs [aria-selected="true"] {
            color: var(--gold) !important;
        }

        div[data-testid="stMetric"] {
            background: #090e16;
            border: 1px solid #202c3a;
            padding: 10px;
            border-radius: 10px;
        }

        div[data-testid="stMetricValue"] {
            color: var(--gold);
        }

        /* Alerts / code / JSON */
        .stAlert {
            background: #0b1119;
            border: 1px solid #263546;
            color: var(--text);
        }

        [data-testid="stJson"] {
            background: #080d14;
            border: 1px solid #202c3a;
            border-radius: 10px;
        }

        hr {
            border-color: #202c39;
        }

        /* Sidebar branding */
        .sidebar-brand {
            color: var(--gold);
            font-weight: 900;
            font-size: 1.25rem;
            letter-spacing: .18em;
        }

        .sidebar-caption {
            color: var(--muted);
            font-size: .72rem;
            line-height: 1.45;
            margin-top: 4px;
        }

        .premium-badge {
            display: inline-block;
            border: 1px solid rgba(245,184,46,.45);
            color: var(--gold);
            background: rgba(245,184,46,.07);
            border-radius: 999px;
            padding: 5px 10px;
            font-size: .68rem;
            font-weight: 800;
            letter-spacing: .12em;
            text-transform: uppercase;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# AUTHENTICATION GUARD
# ---------------------------------------------------------
EXPECTED_PASSWORD = os.getenv("APP_PASSWORD", "admin123")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown(
        """
        <div style="max-width:760px;margin:10vh auto 0;text-align:center;">
            <div class="verdict-brand" style="justify-content:center;">
                <div class="verdict-mark"><span>V</span></div>
                <div class="verdict-word">VERDICT</div>
            </div>
            <div class="verdict-subtitle" style="margin-left:0;">AI INVESTMENT DECISION ENGINE</div>
            <div style="margin-top:28px;" class="premium-badge">PREMIUM MARKET INTELLIGENCE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="text-align:center;color:#8f9bad;margin:18px 0 22px;">Secure access to the VERDICT analysis dashboard.</div>',
        unsafe_allow_html=True,
    )
    pwd_input = st.text_input("Access Key", type="password", label_visibility="collapsed", placeholder="Enter access key")
    if st.button("UNLOCK VERDICT"):
        if pwd_input == EXPECTED_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid Access Key. Check APP_PASSWORD in your .env file.")
    st.stop()

# ---------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="verdict-brand">
        <div class="verdict-mark"><span>V</span></div>
        <div class="verdict-word">VERDICT</div>
    </div>
    <div class="verdict-subtitle">AI INVESTMENT DECISION ENGINE</div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-copy">
        <div class="eyebrow">Premium Crypto Intelligence</div>
        <h1>The market gives you thousands of possibilities.<br><span class="accent">VERDICT gives you one.</span></h1>
        <p>Multi-agent analysis across crypto trajectories, global markets, oil prices, macro indicators, news and current events — followed by historical testing and adaptive learning.</p>
        <div class="market-strip">
            <span class="market-chip">CRYPTO TRAJECTORIES</span>
            <span class="market-chip">GLOBAL MARKETS</span>
            <span class="market-chip">OIL & ENERGY</span>
            <span class="market-chip">MACRO</span>
            <span class="market-chip">NEWS & EVENTS</span>
            <span class="market-chip">ADAPTIVE LEARNING</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# SIDEBAR CONFIGURATION
# ---------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">VERDICT</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-caption">Premium AI investment decision engine</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-kicker">Execution Panel</div>', unsafe_allow_html=True)

    selected_coins = st.multiselect(
        "Crypto assets to scan",
        ["BTC", "ETH", "SOL", "AVAX", "DOT", "LINK"],
        default=["BTC", "ETH", "SOL", "AVAX", "DOT", "LINK"],
    )

    st.markdown("---")
    st.markdown('<div class="section-kicker">System Architecture</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-caption">
        <b style="color:#f5b82e">PASS 1</b> — T−28 historical test<br>
        <b style="color:#20d9ff">PASS 2</b> — T−14 historical test<br>
        <b style="color:#39ff88">PASS 3</b> — Live T+1 decision<br><br>
        GPT-5.5 handles the core reasoning. Lower-cost agents gather, score and train.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown('<div class="premium-badge">PREMIUM</div>', unsafe_allow_html=True)
    st.caption("Decision intelligence — not automated trade execution.")

    if st.button("LOG OUT"):
        st.session_state.authenticated = False
        st.rerun()

# ---------------------------------------------------------
# EXECUTION TRIGGER
# ---------------------------------------------------------
run_analysis = st.button("RUN VERDICT ANALYSIS")

if run_analysis:
    if not selected_coins:
        st.error("Select at least one crypto asset.")
        st.stop()

    learning_dir = PROJECT_ROOT / "predictions_and_learning"
    initial_weights = load_persistent_weights(learning_dir)

    from datetime import datetime, timedelta

    t_minus_28 = (datetime.now() - timedelta(days=28)).strftime("%Y-%m-%d")
    t_minus_14 = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")

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
        "final_recommendation": "",
    }

    app = build_workflow()

    with st.spinner("Running historical tests, adaptive training and GPT-5.5 analysis…"):
        final_state = app.invoke(initial_state)

        mini_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        final_decision = json.loads(final_state["final_recommendation"])

        translation_prompt = f"""
        Summarize this technical recommendation in plain, simple English.
        Target Asset: {final_decision['predicted_best_coin']}
        Rationale: {final_decision['rationale']}
        """
        simple_translation = mini_llm.invoke([SystemMessage(content=translation_prompt)]).content

        st.session_state.final_state = final_state
        st.session_state.final_decision = final_decision
        st.session_state.simple_translation = simple_translation

        st.session_state.pdf_bytes = create_pdf_report(
            target_coin=final_decision["predicted_best_coin"],
            weights=final_state["current_weights"],
            gpt_55_output=final_decision["rationale"],
            gpt_4_translation=simple_translation,
            history=final_state.get("prediction_history", []),
        )

# ---------------------------------------------------------
# RESULTS DISPLAY
# ---------------------------------------------------------
if "final_decision" in st.session_state:
    decision = st.session_state.final_decision
    state = st.session_state.final_state

    st.markdown("---")

    # Decision banner
    st.markdown(
        f"""
        <div class="decision-banner">
            <div>
                <div class="section-kicker">Today's Verdict</div>
                <div class="decision-title">BUY {decision['predicted_best_coin']}</div>
            </div>
            <div style="text-align:right;">
                <div class="premium-badge">GPT-5.5 REASONING</div>
                <div style="color:#8f9bad;font-size:.72rem;margin-top:8px;">Adaptive weights applied</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="section-kicker">Recommended Asset</div>
                <div class="asset-symbol">{decision['predicted_best_coin']}</div>
                <div class="status-pill">High Conviction Allocation</div>
                <div style="margin-top:20px;color:#8f9bad;font-size:.78rem;line-height:1.5;">
                    VERDICT weighs crypto history, global markets, energy prices, macro conditions,
                    news and current events before selecting the strongest opportunity in the scan.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.download_button(
            label="DOWNLOAD PDF REPORT",
            data=st.session_state.pdf_bytes,
            file_name=f"VERDICT_{decision['predicted_best_coin']}_Report.pdf",
            mime="application/pdf",
        )

    with col2:
        st.markdown('<div class="section-kicker">Adaptive Model Weights</div>', unsafe_allow_html=True)
        weights = state["current_weights"]
        cols = st.columns(len(weights))
        for idx, (key, value) in enumerate(weights.items()):
            with cols[idx]:
                st.markdown(
                    f"""
                    <div class="weight-card">
                        <div class="weight-label">{key.replace('_', ' ').title()}</div>
                        <div class="weight-value">{value * 100:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # Feature row
    st.markdown('<div class="section-kicker">What VERDICT Looks At</div>', unsafe_allow_html=True)
    feature_cols = st.columns(6)
    features = [
        ("MULTI-AGENT", "Parallel specialist analysis", "#b66cff"),
        ("40+ SIGNALS", "Market and intelligence inputs", "#20d9ff"),
        ("TRAJECTORIES", "1d / 7d / 14d / 1m / 3m / 6m", "#39ff88"),
        ("GLOBAL MACRO", "Rates, inflation, oil, employment", "#ff9f1c"),
        ("CRYPTO NEWS", "Industry, regulation, events", "#b66cff"),
        ("ADAPTIVE", "Tests outcomes and adjusts weights", "#f5b82e"),
    ]
    for col, (title, copy, accent) in zip(feature_cols, features):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-title" style="color:{accent};">{title}</div>
                    <div class="feature-copy">{copy}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # Output tabs
    tab1, tab2, tab3 = st.tabs(
        ["EXECUTIVE VERDICT", "GPT-5.5 RATIONALE", "MARKET INTELLIGENCE"]
    )

    with tab1:
        st.markdown('<div class="section-kicker">Plain-English Decision</div>', unsafe_allow_html=True)
        st.info(st.session_state.simple_translation)

    with tab2:
        st.markdown('<div class="section-kicker">Deep Reasoning Engine</div>', unsafe_allow_html=True)
        st.markdown(
            f"<div class='panel-card'>{decision['rationale']}</div>",
            unsafe_allow_html=True,
        )

    with tab3:
        st.markdown('<div class="section-kicker">Specialist Market Matrix</div>', unsafe_allow_html=True)
        st.json(state["macro_matrix"])