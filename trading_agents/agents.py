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


# ── Analyst Agents ─────────────────────────────────────────────────────────────

def run_fundamentals_analyst(client: LLMClient, ticker: str) -> str:
    data = "\n\n".join([
        f"=== COMPANY INFO ===\n{get_company_info(ticker)}",
        f"=== FINANCIAL RATIOS & INCOME ===\n{get_financials(ticker)}",
        f"=== BALANCE SHEET ===\n{get_balance_sheet(ticker)}",
        f"=== ANALYST RATINGS ===\n{get_analyst_ratings(ticker)}",
    ])

    system = """You are a CFA-level Fundamentals Analyst at a top-tier investment firm.

Assess the company's intrinsic value using the provided market data.

Produce a comprehensive report covering:
1. Business overview and competitive position
2. Valuation assessment (cheap, fair, or expensive vs peers/history?)
3. Financial health (profitability, margins, growth trajectory)
4. Balance sheet strength (debt load, cash position, liquidity)
5. Analyst consensus and price target implied upside/downside
6. Overall signal: BULLISH / NEUTRAL / BEARISH with key reasoning

Be specific with numbers. Format with clear section headers."""

    return client.chat(
        system,
        f"Conduct a comprehensive fundamental analysis of {ticker.upper()}.\n\n{data}",
        max_tokens=3000,
    )


def run_sentiment_analyst(client: LLMClient, ticker: str) -> str:
    data = "\n\n".join([
        f"=== MARKET SENTIMENT ===\n{get_market_sentiment(ticker)}",
        f"=== RECENT NEWS ===\n{get_news(ticker)}",
        f"=== ANALYST RATINGS ===\n{get_analyst_ratings(ticker)}",
    ])

    system = """You are a Market Sentiment Analyst specializing in crowd psychology and positioning data.

Produce a sentiment report covering:
1. Short interest analysis — heavy bearish positioning? Any short squeeze risk?
2. Institutional/insider ownership trends — smart money moving in or out?
3. News sentiment — is coverage positive, negative, or mixed?
4. Overall market mood around this stock
5. Sentiment signal: BULLISH / NEUTRAL / BEARISH with key reasoning

Interpret data through market psychology and contrarian signals."""

    return client.chat(
        system,
        f"Analyze market sentiment for {ticker.upper()}.\n\n{data}",
        max_tokens=2500,
    )


def run_news_analyst(client: LLMClient, ticker: str) -> str:
    data = "\n\n".join([
        f"=== COMPANY CONTEXT ===\n{get_company_info(ticker)}",
        f"=== NEWS & ARTICLES ===\n{get_news(ticker)}",
    ])

    system = """You are a News & Macro Analyst at a hedge fund.

Produce a news impact report covering:
1. Top 3–5 most important recent news items and their market impact
2. Upcoming catalysts or risk events
3. Macro/sector context and tailwinds/headwinds
4. News signal: BULLISH / NEUTRAL / BEARISH with key reasoning

Be concise; focus on what matters for the next 3–6 months."""

    return client.chat(
        system,
        f"Analyze recent news and catalysts for {ticker.upper()}.\n\n{data}",
        max_tokens=2500,
    )


def run_technical_analyst(client: LLMClient, ticker: str) -> str:
    data = "\n\n".join([
        f"=== TECHNICAL INDICATORS ===\n{get_technical_indicators(ticker)}",
        f"=== PRICE HISTORY (3 months) ===\n{get_price_history(ticker, period='3mo')}",
    ])

    system = """You are a Technical Analyst with 15 years of experience in price action and quantitative signals.

Produce a technical analysis report covering:
1. Trend assessment — uptrend, downtrend, or ranging? (moving average alignment)
2. Momentum signals — RSI and MACD interpretation
3. Key price levels — support, resistance, and potential targets
4. Volume confirmation
5. Technical signal: BULLISH / NEUTRAL / BEARISH with key reasoning

Be specific about price levels and indicator readings."""

    return client.chat(
        system,
        f"Conduct technical analysis of {ticker.upper()}.\n\n{data}",
        max_tokens=2500,
    )


# ── Researcher Agents (debate) ─────────────────────────────────────────────────

def run_bullish_researcher(
    client: LLMClient,
    ticker: str,
    analyst_reports: dict,
    bear_argument: str = "",
    round_num: int = 1,
) -> str:
    reports_text = "\n\n".join(
        f"=== {name} ===\n{report}" for name, report in analyst_reports.items()
    )
    counter = ""
    if bear_argument:
        counter = f"""
The bearish researcher has made the following argument — rebut their key points with evidence:

--- BEAR ARGUMENT (Round {round_num - 1}) ---
{bear_argument}
--- END ---

First rebut the bear's strongest points, then reinforce the bull case."""

    system = """You are the Bullish Researcher at a long/short equity hedge fund.
Your role is to construct the strongest possible BULL case for a stock investment.

You are rigorous, data-driven, and persuasive. Draw on analyst reports to identify
compelling reasons to BUY. Acknowledge risks but explain why upside outweighs them."""

    user = f"""Build the bull case for {ticker.upper()} using these analyst reports:

{reports_text}
{counter}

Provide a structured bull thesis:
1. Core investment thesis (why this stock should appreciate)
2. Key financial/fundamental strengths
3. Technical setup supporting entry
4. Catalysts that could drive the stock higher
5. Rebuttal of the bear case (if applicable)
6. Price target and expected return

Be specific, data-driven, and persuasive."""

    return client.chat(system, user, max_tokens=2500)


def run_bearish_researcher(
    client: LLMClient,
    ticker: str,
    analyst_reports: dict,
    bull_argument: str = "",
    round_num: int = 1,
) -> str:
    reports_text = "\n\n".join(
        f"=== {name} ===\n{report}" for name, report in analyst_reports.items()
    )
    counter = ""
    if bull_argument:
        counter = f"""
The bullish researcher has made the following argument — dismantle their key assumptions:

--- BULL ARGUMENT (Round {round_num}) ---
{bull_argument}
--- END ---

First dismantle the bull's key assumptions, then reinforce the bear case."""

    system = """You are the Bearish Researcher at a long/short equity hedge fund.
Your role is to construct the strongest possible BEAR case — identifying overvaluation,
risks, red flags, and downside scenarios.

You are skeptical, rigorous, and contrarian. Identify what the market might be
missing or overpaying for. Not reflexively negative — analytically rigorous."""

    user = f"""Build the bear case for {ticker.upper()} using these analyst reports:

{reports_text}
{counter}

Provide a structured bear thesis:
1. Core concern / overvaluation risk
2. Financial/fundamental weaknesses or red flags
3. Technical warning signals or unfavorable setup
4. Macro or sector headwinds
5. Rebuttal of the bull case (if applicable)
6. Downside scenario and price target

Be specific, evidence-based, and analytically rigorous."""

    return client.chat(system, user, max_tokens=2500)


# ── Risk Manager ───────────────────────────────────────────────────────────────

def run_risk_manager(
    client: LLMClient,
    ticker: str,
    analyst_reports: dict,
    bull_case: str,
    bear_case: str,
) -> str:
    all_context = "\n\n".join(
        f"=== {name} ===\n{report}" for name, report in analyst_reports.items()
    )

    system = """You are the Chief Risk Officer at a multi-strategy hedge fund.
Your role is to independently assess the risk/reward profile of a proposed trade.

You are NOT making a directional call — you quantify and contextualize risk so the
Portfolio Manager can decide. Focus on: volatility, drawdown potential, liquidity,
concentration risk, macro/tail risks, and position sizing."""

    user = f"""Conduct a risk assessment for a potential position in {ticker.upper()}.

=== ANALYST REPORTS ===
{all_context}

=== BULL CASE ===
{bull_case}

=== BEAR CASE ===
{bear_case}

Provide a risk assessment covering:
1. Volatility profile (historical beta, implied move range)
2. Key risk factors (company-specific, sector, macro)
3. Liquidity assessment (can we enter/exit efficiently?)
4. Suggested position sizing (% of portfolio)
5. Stop-loss level recommendation
6. Risk/reward ratio assessment
7. Overall risk rating: LOW / MEDIUM / HIGH / VERY HIGH"""

    return client.chat(system, user, max_tokens=2000)


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
    all_analyst = "\n\n".join(
        f"=== {name} ===\n{report}" for name, report in analyst_reports.items()
    )
    sep = "=" * 60

    user = f"""You have received comprehensive analysis for {ticker.upper()}.
Make your final trading recommendation.

{sep}
ANALYST REPORTS
{sep}
{all_analyst}

{sep}
BULL CASE
{sep}
{bull_case}

{sep}
BEAR CASE
{sep}
{bear_case}

{sep}
RISK ASSESSMENT
{sep}
{risk_report}

Synthesize all of this into your final recommendation. Be decisive and back your
recommendation with the strongest evidence from the analysis above."""

    yield from client.stream(system_prompt, user, max_tokens=4000)


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
