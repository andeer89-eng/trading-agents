"""
Individual agent implementations.
Each agent runs an agentic tool-use loop and returns a structured report string.
"""

from typing import Generator
import anthropic
from .tools import (
    execute_tool,
    FUNDAMENTALS_TOOLS,
    SENTIMENT_TOOLS,
    NEWS_TOOLS,
    TECHNICAL_TOOLS,
)

MODEL = "claude-opus-4-6"


def _run_tool_loop(
    client: anthropic.Anthropic,
    system: str,
    user_message: str,
    tools: list,
    max_tokens: int = 3000,
) -> str:
    """
    Core agentic loop: run Claude with tools until it produces a final text response.
    Returns the final text output.
    """
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            # Extract text blocks (skip thinking blocks)
            parts = [b.text for b in response.content if b.type == "text"]
            return "\n".join(parts) if parts else "No report generated."

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            # Unexpected stop reason — return whatever text we have
            parts = [b.text for b in response.content if b.type == "text"]
            return "\n".join(parts) if parts else f"Stopped: {response.stop_reason}"


# ─── Analyst Agents ───────────────────────────────────────────────────────────

def run_fundamentals_analyst(client: anthropic.Anthropic, ticker: str) -> str:
    system = """You are a CFA-level Fundamentals Analyst at a top-tier investment firm.

Your job is to assess a company's intrinsic value through rigorous financial analysis.

Use the available tools to gather: company overview, key financial ratios (P/E, PEG, margins,
growth rates), balance sheet health (debt, cash, liquidity), and analyst price targets.

Produce a comprehensive fundamentals report that covers:
1. Business overview and competitive position
2. Valuation assessment (is the stock cheap, fair, or expensive vs peers/history?)
3. Financial health (profitability, margins, growth trajectory)
4. Balance sheet strength (debt load, cash position, liquidity)
5. Analyst consensus and price target implied upside/downside
6. Overall signal: BULLISH / NEUTRAL / BEARISH with key reasoning

Be specific with numbers. Format your report clearly with section headers."""

    return _run_tool_loop(
        client,
        system=system,
        user_message=f"Conduct a comprehensive fundamental analysis of {ticker.upper()}.",
        tools=FUNDAMENTALS_TOOLS,
        max_tokens=3000,
    )


def run_sentiment_analyst(client: anthropic.Anthropic, ticker: str) -> str:
    system = """You are a Market Sentiment Analyst specializing in crowd psychology and
positioning data.

Use the available tools to analyze: short interest (bearish positioning), institutional vs
retail ownership, insider activity, recent news sentiment, and analyst rating trends.

Produce a sentiment report covering:
1. Short interest analysis — are bears heavily positioned? Any short squeeze risk?
2. Institutional/insider ownership trends — smart money moving in or out?
3. News sentiment — is coverage positive, negative, or mixed?
4. Overall market mood around this stock
5. Sentiment signal: BULLISH / NEUTRAL / BEARISH with key reasoning

Interpret the data through the lens of market psychology and contrarian signals."""

    return _run_tool_loop(
        client,
        system=system,
        user_message=f"Analyze market sentiment for {ticker.upper()}.",
        tools=SENTIMENT_TOOLS,
        max_tokens=2500,
    )


def run_news_analyst(client: anthropic.Anthropic, ticker: str) -> str:
    system = """You are a News & Macro Analyst at a hedge fund.

Use the available tools to pull recent news and company context, then assess the impact of:
- Recent company-specific news (earnings, product launches, management changes, legal issues)
- Macro and sector tailwinds/headwinds
- Geopolitical or regulatory risks
- Any event-driven catalysts (upcoming earnings, FDA decisions, contract wins, etc.)

Produce a news impact report covering:
1. Top 3-5 most important recent news items and their market impact
2. Upcoming catalysts or risk events
3. Macro/sector context
4. News signal: BULLISH / NEUTRAL / BEARISH with key reasoning

Be concise and focus on what matters for the next 3-6 months."""

    return _run_tool_loop(
        client,
        system=system,
        user_message=f"Analyze recent news and catalysts for {ticker.upper()}.",
        tools=NEWS_TOOLS,
        max_tokens=2500,
    )


def run_technical_analyst(client: anthropic.Anthropic, ticker: str) -> str:
    system = """You are a Technical Analyst with 15 years of experience in price action and
quantitative signals.

Use the available tools to fetch price history and calculate technical indicators, then analyze:
- Trend structure (moving averages: 20/50/200-day alignment)
- Momentum (RSI: overbought/oversold; MACD: bullish/bearish cross)
- Volatility (Bollinger Bands: squeeze, breakout)
- Volume patterns (confirming or diverging from price)
- Key support and resistance levels

Produce a technical analysis report covering:
1. Trend assessment — is the stock in an uptrend, downtrend, or ranging?
2. Momentum signals — RSI and MACD interpretation
3. Key price levels — support, resistance, and potential targets
4. Volume confirmation
5. Technical signal: BULLISH / NEUTRAL / BEARISH with key reasoning

Be specific about price levels and indicator readings."""

    return _run_tool_loop(
        client,
        system=system,
        user_message=f"Conduct technical analysis of {ticker.upper()}.",
        tools=TECHNICAL_TOOLS,
        max_tokens=2500,
    )


# ─── Researcher Agents (debate) ───────────────────────────────────────────────

def run_bullish_researcher(
    client: anthropic.Anthropic,
    ticker: str,
    analyst_reports: dict,
    bear_argument: str = "",
    round_num: int = 1,
) -> str:
    """Build the bull case, optionally countering the bear argument."""
    reports_text = "\n\n".join(
        f"=== {name} ===\n{report}" for name, report in analyst_reports.items()
    )

    counter_section = ""
    if bear_argument:
        counter_section = f"""
The bearish researcher has made the following argument — you must rebut their key points:

--- BEAR ARGUMENT (Round {round_num - 1}) ---
{bear_argument}
--- END BEAR ARGUMENT ---

In your response, first rebut the bear's strongest points with evidence, then reinforce
the bull case."""

    system = """You are the Bullish Researcher at a long/short equity hedge fund.
Your role is to construct the strongest possible BULL case for a stock investment.

You are rigorous, data-driven, and persuasive. You draw on the analyst reports provided
and identify the most compelling reasons to BUY the stock. You acknowledge risks but
explain why the upside opportunity outweighs them."""

    user_message = f"""Build the bull case for {ticker.upper()} using these analyst reports:

{reports_text}
{counter_section}

Provide a structured bull thesis covering:
1. Core investment thesis (why this stock should appreciate)
2. Key financial/fundamental strengths
3. Technical setup supporting entry
4. Catalysts that could drive the stock higher
5. Rebuttal of the bear case (if applicable)
6. Price target and expected return

Be specific, data-driven, and persuasive."""

    return _run_tool_loop(
        client,
        system=system,
        user_message=user_message,
        tools=[],  # Researchers debate without new data
        max_tokens=2500,
    )


def run_bearish_researcher(
    client: anthropic.Anthropic,
    ticker: str,
    analyst_reports: dict,
    bull_argument: str = "",
    round_num: int = 1,
) -> str:
    """Build the bear case, optionally countering the bull argument."""
    reports_text = "\n\n".join(
        f"=== {name} ===\n{report}" for name, report in analyst_reports.items()
    )

    counter_section = ""
    if bull_argument:
        counter_section = f"""
The bullish researcher has made the following argument — you must rebut their key points:

--- BULL ARGUMENT (Round {round_num}) ---
{bull_argument}
--- END BULL ARGUMENT ---

In your response, first dismantle the bull's key assumptions, then reinforce the bear case."""

    system = """You are the Bearish Researcher at a long/short equity hedge fund.
Your role is to construct the strongest possible BEAR case — identifying overvaluation,
risks, red flags, and downside scenarios.

You are skeptical, rigorous, and contrarian. You identify what the market might be
missing or overpaying for. You are not reflexively negative — you are analytically
rigorous in finding real risks."""

    user_message = f"""Build the bear case for {ticker.upper()} using these analyst reports:

{reports_text}
{counter_section}

Provide a structured bear thesis covering:
1. Core concern (why this stock is risky or overvalued)
2. Financial/fundamental weaknesses or red flags
3. Technical warning signals or unfavorable setup
4. Macro or sector headwinds
5. Rebuttal of the bull case (if applicable)
6. Downside scenario and price target

Be specific, evidence-based, and analytically rigorous."""

    return _run_tool_loop(
        client,
        system=system,
        user_message=user_message,
        tools=[],
        max_tokens=2500,
    )


# ─── Risk Manager ─────────────────────────────────────────────────────────────

def run_risk_manager(
    client: anthropic.Anthropic,
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

You are NOT here to make a directional call — you are here to quantify and contextualize
risk so the Portfolio Manager can make an informed decision. You focus on:
- Volatility and drawdown potential
- Liquidity risk
- Concentration risk
- Macro/tail risks
- Position sizing recommendation"""

    user_message = f"""Conduct a risk assessment for a potential position in {ticker.upper()}.

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
4. Suggested position sizing (% of portfolio) given risk level
5. Stop-loss level recommendation
6. Risk/reward ratio assessment
7. Overall risk rating: LOW / MEDIUM / HIGH / VERY HIGH"""

    return _run_tool_loop(
        client,
        system=system,
        user_message=user_message,
        tools=[],
        max_tokens=2000,
    )


# ─── Portfolio Manager (final synthesis) ──────────────────────────────────────

def run_portfolio_manager(
    client: anthropic.Anthropic,
    ticker: str,
    analyst_reports: dict,
    bull_case: str,
    bear_case: str,
    risk_report: str,
    system_prompt: str,
) -> str:
    """
    The Portfolio Manager synthesizes all inputs and streams the final trading decision.
    Returns the final recommendation as a string.
    """
    all_analyst = "\n\n".join(
        f"=== {name} ===\n{report}" for name, report in analyst_reports.items()
    )

    user_message = f"""You have received comprehensive analysis for {ticker.upper()}.
Make your final trading recommendation.

{'=' * 60}
ANALYST REPORTS
{'=' * 60}
{all_analyst}

{'=' * 60}
BULL CASE (Bullish Researcher)
{'=' * 60}
{bull_case}

{'=' * 60}
BEAR CASE (Bearish Researcher)
{'=' * 60}
{bear_case}

{'=' * 60}
RISK ASSESSMENT (Risk Manager)
{'=' * 60}
{risk_report}
{'=' * 60}

Now synthesize all of this into your final recommendation using the structured output
format specified in your role. Be decisive and back your recommendation with the
strongest evidence from the analysis above."""

    # Stream the portfolio manager's final synthesis
    full_response = []
    print("\n" + "=" * 60)
    print("  PORTFOLIO MANAGER SYNTHESIS")
    print("=" * 60 + "\n")

    with client.messages.stream(
        model=MODEL,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_response.append(text)

    print("\n")
    return "".join(full_response)


def stream_portfolio_manager(
    client: anthropic.Anthropic,
    ticker: str,
    analyst_reports: dict,
    bull_case: str,
    bear_case: str,
    risk_report: str,
    system_prompt: str,
) -> Generator[str, None, None]:
    """
    Generator variant of run_portfolio_manager for use with Streamlit st.write_stream.
    Yields text chunks as they arrive from the Claude streaming API.
    """
    all_analyst = "\n\n".join(
        f"=== {name} ===\n{report}" for name, report in analyst_reports.items()
    )

    user_message = f"""You have received comprehensive analysis for {ticker.upper()}.
Make your final trading recommendation.

{'=' * 60}
ANALYST REPORTS
{'=' * 60}
{all_analyst}

{'=' * 60}
BULL CASE (Bullish Researcher)
{'=' * 60}
{bull_case}

{'=' * 60}
BEAR CASE (Bearish Researcher)
{'=' * 60}
{bear_case}

{'=' * 60}
RISK ASSESSMENT (Risk Manager)
{'=' * 60}
{risk_report}
{'=' * 60}

Now synthesize all of this into your final recommendation using the structured output
format specified in your role. Be decisive and back your recommendation with the
strongest evidence from the analysis above."""

    with client.messages.stream(
        model=MODEL,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            yield text
