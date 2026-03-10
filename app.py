"""
TradingAgents — Streamlit Web Interface
Multi-Agent Financial Trading Framework

Supports Anthropic, OpenAI, Groq, OpenRouter, and any OpenAI-compatible API.
"""

import os
import concurrent.futures
import streamlit as st

from trading_agents.llm import LLMClient, PROVIDERS
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

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TradingAgents",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    .sub-header { color: #888; font-size: 0.95rem; margin-top: 0; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────

col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("## 📈")
with col_title:
    st.markdown('<p class="main-header">TradingAgents</p>', unsafe_allow_html=True)

st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    # ── Stock & debate settings ──────────────────────────────────────────────
    ticker_input = st.text_input(
        "Stock Ticker Symbol",
        value="AAPL",
        placeholder="e.g. AAPL, TSLA, NVDA",
    ).upper().strip()

    debate_rounds = st.select_slider(
        "Debate Rounds",
        options=[1, 2, 3],
        value=2,
        help="More rounds = deeper debate, but slower",
    )

    st.divider()

    # ── AI Provider ──────────────────────────────────────────────────────────
    st.subheader("🤖 AI Provider")

    provider_labels = {k: v["label"] for k, v in PROVIDERS.items()}
    provider_key = st.selectbox(
        "Provider",
        options=list(provider_labels.keys()),
        format_func=lambda k: provider_labels[k],
        index=0,
    )

    pinfo = PROVIDERS[provider_key]

    # Model selector
    if provider_key == "custom":
        selected_model = st.text_input(
            "Model name",
            placeholder="e.g. mistral-small, phi-3-mini",
            help="Enter the exact model name your endpoint expects",
        )
        custom_base_url = st.text_input(
            "Base URL",
            placeholder="http://localhost:11434/v1",
            help="OpenAI-compatible endpoint URL",
        )
    else:
        selected_model = st.selectbox("Model", pinfo["models"], index=0)
        custom_base_url = None

    # API key — check secrets → env var → user input
    env_key = pinfo["key_env"]
    api_key = (
        st.secrets.get(env_key, "")
        or os.environ.get(env_key, "")
    )

    if api_key:
        st.success(f"✓ API key loaded from environment ({env_key})", icon="✅")
    else:
        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder=pinfo["key_hint"],
        )

    st.divider()

    # Validate before enabling button
    ready = bool(ticker_input and api_key and selected_model)
    if provider_key == "custom" and not custom_base_url:
        ready = False

    analyze_btn = st.button(
        "🔍 Analyze Stock",
        type="primary",
        use_container_width=True,
        disabled=not ready,
    )

    st.divider()
    st.caption("**Agent Pipeline:**")
    for cap in [
        "① Fundamentals Analyst",
        "② Sentiment Analyst",
        "③ News Analyst",
        "④ Technical Analyst",
        "⑤ Bullish Researcher",
        "⑥ Bearish Researcher",
        "⑦ Risk Manager",
        "⑧ Portfolio Manager",
    ]:
        st.caption(cap)

# ── Landing state ──────────────────────────────────────────────────────────────

if not analyze_btn:
    st.markdown(
        f'<p class="sub-header">Multi-Agent Financial Trading Framework · '
        f'Provider: <strong>{pinfo["label"]}</strong> · Model: <strong>{selected_model or "—"}</strong></p>',
        unsafe_allow_html=True,
    )
    st.info(
        "👈 Enter a ticker symbol and click **Analyze Stock** to launch the multi-agent pipeline.",
        icon="ℹ️",
    )
    with st.expander("📖 How it works"):
        st.markdown("""
**TradingAgents** deploys a team of 8 specialized AI agents that mirror a real trading firm:

| Phase | Agents | What They Do |
|-------|--------|--------------|
| 1. Intel | Fundamentals, Sentiment, News, Technical | Run **in parallel** to gather market data |
| 2. Debate | Bullish Researcher, Bearish Researcher | Build **opposing theses** through structured debate |
| 3. Risk | Risk Manager | Assess **portfolio risk** and position sizing |
| 4. Decision | Portfolio Manager | **Synthesize** everything into a BUY/SELL/HOLD recommendation |

All analysts use **real-time market data** via yfinance (prices, financials, RSI, MACD, news, etc.)

**Supported providers:** Anthropic · OpenAI · Groq · OpenRouter · Any OpenAI-compatible endpoint
        """)
    st.stop()

# ── Validate ───────────────────────────────────────────────────────────────────

if not api_key:
    st.error("Please provide your API key in the sidebar.")
    st.stop()

if not ticker_input:
    st.error("Please enter a stock ticker symbol.")
    st.stop()

# ── Create client ──────────────────────────────────────────────────────────────

try:
    client = LLMClient(
        provider=provider_key,
        api_key=api_key,
        model=selected_model,
        base_url=custom_base_url,
    )
except Exception as e:
    st.error(f"Failed to initialize AI client: {e}")
    st.stop()

# ── Ticker banner ──────────────────────────────────────────────────────────────

st.markdown(f"## 📊 Analysis: `{ticker_input}`")
st.caption(f"Provider: {pinfo['label']} · Model: {selected_model}")

# ── STEP 1: Parallel Analyst Team ─────────────────────────────────────────────

st.markdown("### Step 1 — Analyst Team")
analyst_reports: dict[str, str] = {}

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
        status_placeholders[name].markdown(f"{analyst_icons[name]} **{name}**\n\n⏳ Waiting...")

def _run_analyst(args):
    name, fn = args
    try:
        return name, fn(client, ticker_input), None
    except Exception as e:
        return name, None, str(e)

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(_run_analyst, (n, f)): n for n, f in analyst_fns.items()}
    for future in concurrent.futures.as_completed(futures):
        name, report, error = future.result()
        if error:
            analyst_reports[name] = "[Analysis unavailable]"
            status_placeholders[name].markdown(f"{analyst_icons[name]} **{name}**\n\n❌ Failed")
            st.error(f"**{name} failed:** {error}")
        else:
            analyst_reports[name] = report
            status_placeholders[name].markdown(f"{analyst_icons[name]} **{name}**\n\n✅ Complete")

if all(r.startswith("[Analysis") for r in analyst_reports.values()):
    st.error("All analysts failed. Check your API key and model name.")
    st.stop()

st.markdown("**Analyst Reports:**")
tabs = st.tabs([f"{analyst_icons[n]} {n}" for n in analyst_reports])
for tab, (name, report) in zip(tabs, analyst_reports.items()):
    with tab:
        st.markdown(report)

st.divider()

# ── STEP 2: Researcher Debate ──────────────────────────────────────────────────

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

# ── STEP 3: Risk Assessment ────────────────────────────────────────────────────

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

# ── STEP 4: Portfolio Manager Final Recommendation (streamed) ──────────────────

st.markdown("### Step 4 — Portfolio Manager Recommendation")
st.caption("Synthesizing all intelligence into a final trading decision (streaming)...")

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
    "AI-generated for educational purposes only — not financial advice.",
    icon="✅",
)

report_text = f"""# TradingAgents Report: {ticker_input}

Provider: {pinfo['label']} · Model: {selected_model}

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
)
