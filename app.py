"""
TradingAgents — Streamlit Web Interface
Multi-Agent Financial Trading Framework powered by Claude claude-opus-4-6 + yfinance
"""

import os
import concurrent.futures
import streamlit as st
import anthropic

from trading_agents.agents import (
    run_fundamentals_analyst,
    run_sentiment_analyst,
    run_news_analyst,
    run_technical_analyst,
    run_bullish_researcher,
    run_bearish_researcher,
    run_risk_manager,
    stream_portfolio_manager,
)
from trading_agents.orchestrator import PORTFOLIO_MANAGER_SYSTEM

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TradingAgents",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4aa, #0084ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #888;
        font-size: 0.95rem;
        margin-top: 0;
    }
    .metric-card {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #333;
    }
    .agent-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────

col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("## 📈")
with col_title:
    st.markdown('<p class="main-header">TradingAgents</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Multi-Agent Financial Trading Framework · '
        'Powered by Claude claude-opus-4-6 + yfinance</p>',
        unsafe_allow_html=True,
    )

st.divider()

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    ticker_input = st.text_input(
        "Stock Ticker Symbol",
        value="AAPL",
        placeholder="e.g. AAPL, TSLA, NVDA",
        help="Enter any valid US stock ticker symbol",
    ).upper().strip()

    debate_rounds = st.select_slider(
        "Debate Rounds",
        options=[1, 2, 3],
        value=2,
        help="More rounds = deeper bull/bear debate, but slower",
    )

    st.divider()
    st.subheader("🔑 API Key")

    # Try: Streamlit secrets → env var → user input
    api_key = (
        st.secrets.get("ANTHROPIC_API_KEY", "")
        or os.environ.get("ANTHROPIC_API_KEY", "")
    )

    if api_key:
        st.success("✓ API key loaded from environment", icon="✅")
    else:
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-...",
            help="Get your key at console.anthropic.com",
        )

    st.divider()
    analyze_btn = st.button(
        "🔍 Analyze Stock",
        type="primary",
        use_container_width=True,
        disabled=not (ticker_input and api_key),
    )

    st.divider()
    st.caption("**Agent Pipeline:**")
    st.caption("① Fundamentals Analyst")
    st.caption("② Sentiment Analyst")
    st.caption("③ News Analyst")
    st.caption("④ Technical Analyst")
    st.caption("⑤ Bullish Researcher")
    st.caption("⑥ Bearish Researcher")
    st.caption("⑦ Risk Manager")
    st.caption("⑧ Portfolio Manager")

# ─── Landing state ────────────────────────────────────────────────────────────

if not analyze_btn:
    st.info(
        "👈 Enter a ticker symbol and click **Analyze Stock** to launch the multi-agent pipeline.",
        icon="ℹ️",
    )
    with st.expander("📖 How it works"):
        st.markdown("""
**TradingAgents** deploys a team of 8 specialized Claude AI agents that mirror a real trading firm:

| Phase | Agents | What They Do |
|-------|--------|--------------|
| 1. Intel | Fundamentals, Sentiment, News, Technical | Run **in parallel** to gather market data |
| 2. Debate | Bullish Researcher, Bearish Researcher | Build **opposing theses** through structured debate |
| 3. Risk | Risk Manager | Assess **portfolio risk** and position sizing |
| 4. Decision | Portfolio Manager | **Synthesize** everything into a BUY/SELL/HOLD recommendation |

All analysts use **real-time market data** via yfinance (prices, financials, RSI, MACD, news, etc.)
        """)
    st.stop()

# ─── Validate inputs ──────────────────────────────────────────────────────────

if not api_key:
    st.error("Please provide your Anthropic API key in the sidebar.")
    st.stop()

if not ticker_input:
    st.error("Please enter a stock ticker symbol.")
    st.stop()

# ─── Initialize client ────────────────────────────────────────────────────────

try:
    client = anthropic.Anthropic(api_key=api_key)
except Exception as e:
    st.error(f"Failed to initialize Anthropic client: {e}")
    st.stop()

# ─── Ticker banner ────────────────────────────────────────────────────────────

st.markdown(f"## 📊 Analysis: `{ticker_input}`")

# ─── STEP 1: Parallel Analyst Team ───────────────────────────────────────────

st.markdown("### Step 1 — Analyst Team")
analyst_reports = {}

analyst_fns = {
    "Fundamentals Analyst": run_fundamentals_analyst,
    "Sentiment Analyst":    run_sentiment_analyst,
    "News Analyst":         run_news_analyst,
    "Technical Analyst":    run_technical_analyst,
}

analyst_icons = {
    "Fundamentals Analyst": "💰",
    "Sentiment Analyst":    "🧠",
    "News Analyst":         "📰",
    "Technical Analyst":    "📉",
}

progress_cols = st.columns(4)
status_placeholders = {}
for i, name in enumerate(analyst_fns):
    with progress_cols[i]:
        status_placeholders[name] = st.empty()
        status_placeholders[name].markdown(
            f"{analyst_icons[name]} **{name}**\n\n⏳ Waiting..."
        )

def run_analyst_with_status(args):
    name, fn = args
    try:
        report = fn(client, ticker_input)
        return name, report, None
    except Exception as e:
        return name, None, str(e)

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = {
        executor.submit(run_analyst_with_status, (name, fn)): name
        for name, fn in analyst_fns.items()
    }
    for future in concurrent.futures.as_completed(futures):
        name, report, error = future.result()
        if error:
            analyst_reports[name] = f"[Analysis unavailable — see error below]"
            status_placeholders[name].markdown(
                f"{analyst_icons[name]} **{name}**\n\n❌ Failed"
            )
            st.error(f"**{name} failed:** {error}")
        else:
            analyst_reports[name] = report
            status_placeholders[name].markdown(
                f"{analyst_icons[name]} **{name}**\n\n✅ Complete"
            )

if all(r.startswith("[Analysis unavailable") for r in analyst_reports.values()):
    st.error("All analyst agents failed. Check the errors above — your API key may lack access to this model.")
    st.stop()

# Show analyst reports in tabs
st.markdown("**Analyst Reports** (click to expand):")
tab_labels = [f"{analyst_icons[n]} {n}" for n in analyst_reports]
tabs = st.tabs(tab_labels)
for tab, (name, report) in zip(tabs, analyst_reports.items()):
    with tab:
        st.markdown(report)

st.divider()

# ─── STEP 2: Researcher Debate ────────────────────────────────────────────────

st.markdown("### Step 2 — Researcher Debate")
st.caption(f"Running {debate_rounds} round(s) of bull vs bear debate...")

bull_argument = ""
bear_argument = ""

for round_num in range(1, debate_rounds + 1):
    with st.status(f"🥊 Debate Round {round_num}/{debate_rounds}", expanded=False) as debate_status:
        try:
            st.write("Bullish Researcher building case...")
            bull_argument = run_bullish_researcher(
                client, ticker_input, analyst_reports, bear_argument, round_num
            )
            st.write("✅ Bull case ready")

            st.write("Bearish Researcher countering...")
            bear_argument = run_bearish_researcher(
                client, ticker_input, analyst_reports, bull_argument, round_num
            )
            st.write("✅ Bear case ready")
            debate_status.update(label=f"✅ Round {round_num} complete", state="complete")
        except Exception as e:
            debate_status.update(label=f"❌ Round {round_num} failed", state="error")
            st.error(f"Researcher debate failed: {e}")
            st.stop()

col_bull, col_bear = st.columns(2)
with col_bull:
    with st.expander("🟢 Bull Case", expanded=True):
        st.markdown(bull_argument)
with col_bear:
    with st.expander("🔴 Bear Case", expanded=True):
        st.markdown(bear_argument)

st.divider()

# ─── STEP 3: Risk Assessment ──────────────────────────────────────────────────

st.markdown("### Step 3 — Risk Assessment")

with st.status("⚠️ Risk Manager assessing position risk...", expanded=False) as risk_status:
    try:
        risk_report = run_risk_manager(
            client, ticker_input, analyst_reports, bull_argument, bear_argument
        )
        risk_status.update(label="✅ Risk assessment complete", state="complete")
    except Exception as e:
        risk_status.update(label="❌ Risk assessment failed", state="error")
        st.error(f"Risk Manager failed: {e}")
        st.stop()

with st.expander("📋 Risk Report"):
    st.markdown(risk_report)

st.divider()

# ─── STEP 4: Portfolio Manager Final Recommendation (streamed) ────────────────

st.markdown("### Step 4 — Portfolio Manager Recommendation")
st.caption("Synthesizing all intelligence into a final trading decision (streaming)...")

recommendation_container = st.empty()

full_recommendation = st.write_stream(
    stream_portfolio_manager(
        client=client,
        ticker=ticker_input,
        analyst_reports=analyst_reports,
        bull_case=bull_argument,
        bear_case=bear_argument,
        risk_report=risk_report,
        system_prompt=PORTFOLIO_MANAGER_SYSTEM,
    )
)

st.divider()
st.success(
    f"✅ Analysis complete for **{ticker_input}**. "
    "This is AI-generated analysis for educational purposes only — not financial advice.",
    icon="✅",
)

# Download button for the full report
report_text = f"""# TradingAgents Report: {ticker_input}

## Analyst Reports

{chr(10).join(f"### {name}{chr(10)}{report}" for name, report in analyst_reports.items())}

## Bull Case
{bull_argument}

## Bear Case
{bear_argument}

## Risk Assessment
{risk_report}

## Portfolio Manager Recommendation
{full_recommendation}

---
*Generated by TradingAgents — AI-powered analysis for educational purposes only.*
*Not financial advice. Always do your own research.*
"""

st.download_button(
    label="📥 Download Full Report",
    data=report_text,
    file_name=f"trading_report_{ticker_input}.md",
    mime="text/markdown",
    use_container_width=False,
)
