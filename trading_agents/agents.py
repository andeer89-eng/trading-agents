"""
Individual agent implementations.

Each agent pre-fetches the market data it needs via tools.py, then calls the
LLM with that data as context.  This approach works with any provider
(Anthropic, OpenAI, Groq, OpenRouter, or any OpenAI-compatible endpoint)
because it requires only a standard chat-completion interface — no tool-use
loop needed.
"""

from __future__ import annotations

from typing import Generator

from .llm import LLMClient
from .tools import (
    get_company_info,
    get_financials,
    get_balance_sheet,
    get_analyst_ratings,
    get_market_sentiment,
    get_news,
    get_technical_indicators,
    get_price_history,
)


def _condense_reports(analyst_reports: dict, max_chars: int = 600) -> str:
    """Truncate analyst reports to keep downstream prompts short."""
    lines = []
    for name, report in analyst_reports.items():
        snippet = report[:max_chars].strip()
        if len(report) > max_chars:
            snippet += "…"
        lines.append(f"[{name}]\n{snippet}")
    return "\n\n".join(lines)


# ── Analyst Agents ─────────────────────────────────────────────────────────────

def run_fundamentals_analyst(client: LLMClient, ticker: str) -> str:
    data = "\n\n".join([
        f"=== COMPANY INFO ===\n{get_company_info(ticker)}",
        f"=== FINANCIAL RATIOS & INCOME ===\n{get_financials(ticker)}",
        f"=== BALANCE SHEET ===\n{get_balance_sheet(ticker)}",
        f"=== ANALYST RATINGS ===\n{get_analyst_ratings(ticker)}",
    ])

    system = """You are a CFA-level Fundamentals Analyst. Be concise and data-driven.

Produce a SHORT structured report using bullet points:
• **Valuation** — cheap / fair / expensive vs peers (cite 2-3 key multiples)
• **Financial Health** — margins, growth, profitability (1-2 bullets)
• **Balance Sheet** — debt, cash, liquidity (1 bullet)
• **Analyst Consensus** — rating, price target, implied upside (1 bullet)
• **Signal: BULLISH / NEUTRAL / BEARISH** — one-sentence reason

Max 300 words. Numbers only where relevant."""

    return client.chat(
        system,
        f"Fundamental analysis of {ticker.upper()}.\n\n{data}",
        max_tokens=700,
    )


def run_sentiment_analyst(client: LLMClient, ticker: str) -> str:
    data = "\n\n".join([
        f"=== MARKET SENTIMENT ===\n{get_market_sentiment(ticker)}",
        f"=== RECENT NEWS ===\n{get_news(ticker)}",
        f"=== ANALYST RATINGS ===\n{get_analyst_ratings(ticker)}",
    ])

    system = """You are a Market Sentiment Analyst. Be concise and data-driven.

Produce a SHORT structured report using bullet points:
• **Short Interest** — bearish positioning or squeeze risk (1 bullet)
• **Ownership Trends** — institutions/insiders moving in or out (1 bullet)
• **News Sentiment** — positive, negative, or mixed (1 bullet)
• **Signal: BULLISH / NEUTRAL / BEARISH** — one-sentence reason

Max 200 words."""

    return client.chat(
        system,
        f"Sentiment analysis for {ticker.upper()}.\n\n{data}",
        max_tokens=500,
    )


def run_news_analyst(client: LLMClient, ticker: str) -> str:
    data = "\n\n".join([
        f"=== COMPANY CONTEXT ===\n{get_company_info(ticker)}",
        f"=== NEWS & ARTICLES ===\n{get_news(ticker)}",
    ])

    system = """You are a News & Catalyst Analyst. Be concise.

Produce a SHORT structured report using bullet points:
• **Top 3 News Items** — title + one-line impact each
• **Upcoming Catalysts** — 1-2 bullets
• **Macro/Sector Context** — 1 bullet
• **Signal: BULLISH / NEUTRAL / BEARISH** — one-sentence reason

Max 200 words."""

    return client.chat(
        system,
        f"News and catalyst analysis for {ticker.upper()}.\n\n{data}",
        max_tokens=500,
    )


def run_technical_analyst(client: LLMClient, ticker: str) -> str:
    data = "\n\n".join([
        f"=== TECHNICAL INDICATORS ===\n{get_technical_indicators(ticker)}",
        f"=== PRICE HISTORY (3 months) ===\n{get_price_history(ticker, period='3mo')}",
    ])

    system = """You are a Technical Analyst. Be concise and specific with price levels.

Produce a SHORT structured report using bullet points:
• **Trend** — uptrend / downtrend / ranging (cite MA alignment)
• **Momentum** — RSI and MACD reading (specific numbers)
• **Key Levels** — support and resistance prices
• **Volume** — confirming or diverging
• **Signal: BULLISH / NEUTRAL / BEARISH** — one-sentence reason

Max 200 words."""

    return client.chat(
        system,
        f"Technical analysis of {ticker.upper()}.\n\n{data}",
        max_tokens=500,
    )


# ── Researcher Agents (debate) ─────────────────────────────────────────────────

def run_bullish_researcher(
    client: LLMClient,
    ticker: str,
    analyst_reports: dict,
    bear_argument: str = "",
    round_num: int = 1,
) -> str:
    summary = _condense_reports(analyst_reports, max_chars=500)
    counter = ""
    if bear_argument:
        bear_snippet = bear_argument[:600].strip()
        if len(bear_argument) > 600:
            bear_snippet += "…"
        counter = f"\n\nBEAR ARGUMENT TO REBUT:\n{bear_snippet}\nRebut the 2 strongest bear points, then reinforce the bull case."

    system = """You are the Bullish Researcher at a hedge fund. Be concise and data-driven.

Build the bull case in bullet-point format:
• **Core Thesis** — why this stock should appreciate (2 bullets max)
• **Key Strengths** — financial or fundamental (2 bullets)
• **Catalysts** — what could drive it higher (2 bullets)
• **Rebuttal** — address bear concerns if applicable (1-2 bullets)
• **Price Target** — expected return

Max 250 words."""

    user = f"Bull case for {ticker.upper()}:\n\n{summary}{counter}"
    return client.chat(system, user, max_tokens=700)


def run_bearish_researcher(
    client: LLMClient,
    ticker: str,
    analyst_reports: dict,
    bull_argument: str = "",
    round_num: int = 1,
) -> str:
    summary = _condense_reports(analyst_reports, max_chars=500)
    counter = ""
    if bull_argument:
        bull_snippet = bull_argument[:600].strip()
        if len(bull_argument) > 600:
            bull_snippet += "…"
        counter = f"\n\nBULL ARGUMENT TO COUNTER:\n{bull_snippet}\nDismantle the 2 strongest bull points, then reinforce the bear case."

    system = """You are the Bearish Researcher at a hedge fund. Be concise and analytically rigorous.

Build the bear case in bullet-point format:
• **Core Risk** — overvaluation or key concern (2 bullets max)
• **Weaknesses** — financial or fundamental red flags (2 bullets)
• **Headwinds** — macro or sector risks (1-2 bullets)
• **Rebuttal** — dismantle bull assumptions if applicable (1-2 bullets)
• **Downside Target** — expected drawdown

Max 250 words."""

    user = f"Bear case for {ticker.upper()}:\n\n{summary}{counter}"
    return client.chat(system, user, max_tokens=700)


# ── Risk Manager ───────────────────────────────────────────────────────────────

def run_risk_manager(
    client: LLMClient,
    ticker: str,
    analyst_reports: dict,
    bull_case: str,
    bear_case: str,
) -> str:
    summary = _condense_reports(analyst_reports, max_chars=400)
    bull_snippet = bull_case[:400].strip()
    bear_snippet = bear_case[:400].strip()

    system = """You are the Chief Risk Officer. Be concise.

Risk assessment in bullet-point format:
• **Volatility** — beta, historical range (1 bullet)
• **Key Risks** — top 3 risks (company, sector, macro)
• **Liquidity** — can we enter/exit efficiently? (1 bullet)
• **Position Sizing** — suggested % of portfolio
• **Stop-Loss** — recommended level
• **Risk Rating: LOW / MEDIUM / HIGH / VERY HIGH**

Max 200 words."""

    user = f"""Risk assessment for {ticker.upper()}.

ANALYST SUMMARY:
{summary}

BULL CASE: {bull_snippet}

BEAR CASE: {bear_snippet}"""

    return client.chat(system, user, max_tokens=500)


# ── Portfolio Manager ──────────────────────────────────────────────────────────

def stream_portfolio_manager(
    client: LLMClient,
    ticker: str,
    analyst_reports: dict,
    bull_case: str,
    bear_case: str,
    risk_report: str,
    system_prompt: str,
) -> Generator[str, None, None]:
    """Generator variant for use with Streamlit st.write_stream; yields text chunks."""
    summary = _condense_reports(analyst_reports, max_chars=400)

    user = f"""Final trading recommendation for {ticker.upper()}.

ANALYST SIGNALS:
{summary}

BULL CASE:
{bull_case[:600]}

BEAR CASE:
{bear_case[:600]}

RISK ASSESSMENT:
{risk_report[:500]}

Provide a decisive recommendation in this format:
## Recommendation
**Action**: BUY / SELL / HOLD
**Conviction**: HIGH / MEDIUM / LOW
**Time Horizon**: (e.g. 3–6 months)

## Rationale
3-5 bullet points synthesizing the key evidence.

## Key Risks
2-3 bullet points.

## Position Sizing
One sentence."""

    yield from client.stream(system_prompt, user, max_tokens=800)


def run_portfolio_manager(
    client: LLMClient,
    ticker: str,
    analyst_reports: dict,
    bull_case: str,
    bear_case: str,
    risk_report: str,
    system_prompt: str,
) -> str:
    """Non-streaming variant used by the CLI orchestrator."""
    return "".join(
        stream_portfolio_manager(
            client, ticker, analyst_reports, bull_case, bear_case, risk_report, system_prompt
        )
    )
