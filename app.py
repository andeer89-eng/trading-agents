"""
TradingAgents — Streamlit Web Interface
Multi-Agent Financial Trading Framework

Supports Anthropic, OpenAI, Groq, OpenRouter, and any OpenAI-compatible API.
"""

import io
import os
import concurrent.futures
import streamlit as st

from trading_agents.llm import LLMClient, PROVIDERS
from trading_agents.tools import configure_eulerpool
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
    .history-item {
        padding: 6px 10px;
        border-radius: 6px;
        border: 1px solid #333;
        margin-bottom: 4px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────

if "scan_history" not in st.session_state:
    st.session_state.scan_history = []  # list of dicts: {ticker, ts, provider, model, signal, report}
if "viewed_report" not in st.session_state:
    st.session_state.viewed_report = None  # index of history entry being viewed

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
        value=1,
        help="1 round = fast & cheap (~$0.10). More rounds = deeper debate but higher cost.",
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

    # ── Eulerpool API (optional) ──────────────────────────────────────────────
    st.subheader("📊 Market Data")
    st.caption("Eulerpool provides richer fundamental data (free tier available).")

    euler_key = (
        st.secrets.get("EULERPOOL_API_KEY", "")
        or os.environ.get("EULERPOOL_API_KEY", "")
    )
    if euler_key:
        st.success("✓ Eulerpool key loaded from environment", icon="✅")
    else:
        euler_key = st.text_input(
            "Eulerpool API Key (optional)",
            type="password",
            placeholder="ep_xxxxxxxxxxxxxxxx",
            help="Leave blank to use yfinance only. Get a free key at eulerpool.com",
        )

    if euler_key:
        configure_eulerpool(euler_key)
        st.caption("Data source: Eulerpool → yfinance fallback")
    else:
        st.caption("Data source: yfinance")

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

    # ── Scan history ──────────────────────────────────────────────────────────
    if st.session_state.scan_history:
        st.divider()
        st.subheader("🕘 Scan History")
        history = st.session_state.scan_history
        for idx in range(len(history) - 1, max(len(history) - 31, -1), -1):
            entry = history[idx]
            signal_icon = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(
                entry.get("signal", "").upper(), "⚪"
            )
            st.caption(
                f"{signal_icon} **{entry['ticker']}** · {entry['ts']}\n\n"
                f"{entry['provider']} / {entry['model']}"
            )
            if entry.get("report"):
                if st.button("📄 View Report", key=f"view_{idx}"):
                    st.session_state.viewed_report = idx
                    st.rerun()

# ── Saved report view ──────────────────────────────────────────────────────────

if not analyze_btn and st.session_state.viewed_report is not None:
    entry = st.session_state.scan_history[st.session_state.viewed_report]
    r = entry["report"]
    signal_icon = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(entry.get("signal", "").upper(), "⚪")
    st.markdown(f"## 📊 Saved Report: `{entry['ticker']}`")
    st.caption(f"{entry['provider']} / {entry['model']} · {entry['ts']} · {signal_icon} **{entry.get('signal', '—')}**")
    if st.button("✖ Close Report"):
        st.session_state.viewed_report = None
        st.rerun()
    st.divider()
    analyst_icons = {"Fundamentals Analyst": "💰", "Sentiment Analyst": "🧠",
                     "News Analyst": "📰", "Technical Analyst": "📉"}
    with st.expander("📋 Analyst Reports", expanded=False):
        tabs = st.tabs([f"{analyst_icons.get(n, '📊')} {n}" for n in r["analyst_reports"]])
        for tab, (name, report) in zip(tabs, r["analyst_reports"].items()):
            with tab:
                st.markdown(report)
    col_bull, col_bear = st.columns(2)
    with col_bull:
        with st.expander("🟢 Bull Case", expanded=False):
            st.markdown(r["bull_argument"])
    with col_bear:
        with st.expander("🔴 Bear Case", expanded=False):
            st.markdown(r["bear_argument"])
    with st.expander("⚠️ Risk Report", expanded=False):
        st.markdown(r["risk_report"])
    st.markdown("### Portfolio Manager Recommendation")
    st.markdown(r["full_recommendation"])
    st.stop()

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

All analysts use **real-time market data** via Eulerpool (if key provided) or yfinance.

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

with st.expander("📋 Analyst Reports", expanded=False):
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
    with st.expander("🟢 Bull Case", expanded=False):
        st.markdown(bull_argument)
with col_bear:
    with st.expander("🔴 Bear Case", expanded=False):
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

with st.expander("⚠️ Risk Report", expanded=False):
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

# ── Record scan in history ─────────────────────────────────────────────────────

import re
from datetime import datetime as _dt

# Best-effort signal extraction from the recommendation text
_signal_match = re.search(r'\*\*Action\*\*:\s*(BUY|SELL|HOLD)', full_recommendation, re.IGNORECASE)
_signal = _signal_match.group(1).upper() if _signal_match else "—"

st.session_state.scan_history.append({
    "ticker": ticker_input,
    "ts": _dt.now().strftime("%Y-%m-%d %H:%M"),
    "provider": pinfo["label"],
    "model": selected_model,
    "signal": _signal,
    "report": {
        "analyst_reports": analyst_reports,
        "bull_argument": bull_argument,
        "bear_argument": bear_argument,
        "risk_report": risk_report,
        "full_recommendation": full_recommendation,
    },
})
# Keep at most 30 entries
st.session_state.scan_history = st.session_state.scan_history[-30:]

st.divider()
st.success(
    f"✅ Analysis complete for **{ticker_input}** — Recommendation: **{_signal}**. "
    "AI-generated for educational purposes only — not financial advice.",
    icon="✅",
)

# ── Build full report text ─────────────────────────────────────────────────────

report_text = f"""# TradingAgents Report: {ticker_input}

Provider: {pinfo['label']} · Model: {selected_model}
Generated: {_dt.now().strftime("%Y-%m-%d %H:%M")}

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

# ── Download buttons ──────────────────────────────────────────────────────────

dl_col1, dl_col2 = st.columns(2)

with dl_col1:
    st.download_button(
        label="📥 Download Report (Markdown)",
        data=report_text,
        file_name=f"trading_report_{ticker_input}.md",
        mime="text/markdown",
    )

with dl_col2:
    # Generate PDF using reportlab (if available) or fallback to HTML
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
        from reportlab.lib import colors

        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=18, spaceAfter=12)
        h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=6)
        body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9, leading=13, spaceAfter=4)

        story = []
        story.append(Paragraph(f"TradingAgents Report: {ticker_input}", title_style))
        story.append(Paragraph(f"Provider: {pinfo['label']} · Model: {selected_model}", body_style))
        story.append(Paragraph(f"Generated: {_dt.now().strftime('%Y-%m-%d %H:%M')}", body_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.grey, spaceAfter=10))

        import re as _re

        h3_style = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=10, spaceBefore=8, spaceAfter=4)
        bullet_style = ParagraphStyle("Bullet", parent=styles["Normal"], fontSize=9, leading=13,
                                      leftIndent=12, spaceAfter=2)

        def _clean(text: str) -> str:
            """Convert inline markdown to ReportLab XML."""
            # Bold+italic ***text*** or **text**
            text = _re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i>\1</i></b>', text)
            text = _re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
            text = _re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
            # Escape bare & and < that aren't already tags
            text = _re.sub(r'&(?!amp;|lt;|gt;|#)', '&amp;', text)
            return text

        def _add_section(heading: str, content: str):
            story.append(Paragraph(heading, h2_style))
            for line in content.split("\n"):
                raw = line.rstrip()
                # Skip horizontal rules
                if _re.match(r'^-{3,}$', raw) or _re.match(r'^\*{3,}$', raw):
                    story.append(HRFlowable(width="100%", thickness=0.5,
                                            color=colors.lightgrey, spaceAfter=4))
                    continue
                # H3 headers (###)
                m = _re.match(r'^#{2,3}\s+(.*)', raw)
                if m:
                    story.append(Paragraph(_clean(m.group(1)), h3_style))
                    continue
                # H2 headers (##) already handled above, treat remaining # as h3
                m = _re.match(r'^#\s+(.*)', raw)
                if m:
                    story.append(Paragraph(_clean(m.group(1)), h3_style))
                    continue
                # Table rows  | col | col | — render as indented text, skip separator rows
                if raw.startswith("|"):
                    if _re.match(r'^\|[\s\-|]+\|$', raw):
                        continue  # skip |---|---| rows
                    cells = [c.strip() for c in raw.strip("|").split("|")]
                    cell_text = "  ·  ".join(_clean(c) for c in cells if c)
                    if cell_text:
                        try:
                            story.append(Paragraph(cell_text, bullet_style))
                        except Exception:
                            pass
                    continue
                # Bullet lines
                m = _re.match(r'^[-•*]\s+(.*)', raw)
                if m:
                    try:
                        story.append(Paragraph("• " + _clean(m.group(1)), bullet_style))
                    except Exception:
                        pass
                    continue
                # Normal text
                clean = _clean(raw).strip()
                if clean:
                    try:
                        story.append(Paragraph(clean, body_style))
                    except Exception:
                        pass
            story.append(Spacer(1, 8))

        for name, report in analyst_reports.items():
            _add_section(name, report)

        _add_section("Bull Case", bull_argument)
        _add_section("Bear Case", bear_argument)
        _add_section("Risk Assessment", risk_report)
        _add_section("Portfolio Manager Recommendation", full_recommendation)

        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey, spaceAfter=6))
        story.append(Paragraph("AI-generated for educational purposes only. Not financial advice.", body_style))

        doc.build(story)
        pdf_bytes = pdf_buffer.getvalue()

        st.download_button(
            label="📄 Download Report (PDF)",
            data=pdf_bytes,
            file_name=f"trading_report_{ticker_input}.pdf",
            mime="application/pdf",
        )
    except ImportError:
        # reportlab not installed — offer HTML instead
        html_report = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>TradingAgents Report: {ticker_input}</title>
<style>
  body {{ font-family: sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; line-height:1.6; }}
  h1 {{ color: #0084ff; }} h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 4px; margin-top:2rem; }}
  h3 {{ color: #333; margin-top:1.2rem; }}
  .md {{ white-space: pre-wrap; font-family: inherit; }}
  strong {{ font-weight: 700; }}
</style></head><body>
<h1>TradingAgents Report: {ticker_input}</h1>
<p><strong>Provider:</strong> {pinfo['label']} · <strong>Model:</strong> {selected_model}<br>
<strong>Generated:</strong> {_dt.now().strftime('%Y-%m-%d %H:%M')}</p>
<hr>
{''.join(f"<h2>{name}</h2><div class='md'>{report}</div>" for name, report in analyst_reports.items())}
<h2>Bull Case</h2><div class='md'>{bull_argument}</div>
<h2>Bear Case</h2><div class='md'>{bear_argument}</div>
<h2>Risk Assessment</h2><div class='md'>{risk_report}</div>
<h2>Portfolio Manager Recommendation</h2><div class='md'>{full_recommendation}</div>
<hr><p><em>AI-generated for educational purposes only. Not financial advice.</em></p>
</body></html>"""
        st.download_button(
            label="📄 Download Report (HTML)",
            data=html_report,
            file_name=f"trading_report_{ticker_input}.html",
            mime="text/html",
        )
