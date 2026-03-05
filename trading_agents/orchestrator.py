"""
Multi-agent trading workflow orchestrator.

Execution order:
  1. Analyst team (parallel): Fundamentals, Sentiment, News, Technical
  2. Researcher debate (sequential rounds): Bullish ↔ Bearish
  3. Risk Manager assessment
  4. Portfolio Manager final synthesis (streamed)
"""

import concurrent.futures
import time
import anthropic

from .agents import (
    run_fundamentals_analyst,
    run_sentiment_analyst,
    run_news_analyst,
    run_technical_analyst,
    run_bullish_researcher,
    run_bearish_researcher,
    run_risk_manager,
    run_portfolio_manager,
)

# The Portfolio Manager system prompt (from the user's specification)
PORTFOLIO_MANAGER_SYSTEM = """You are the **Portfolio Manager & Trading Coordinator** of a sophisticated multi-agent trading system that mirrors real-world trading firms. Your role is to orchestrate specialized analyst and researcher agents to collaboratively evaluate market conditions and make informed trading decisions.

## Your Role

As the Portfolio Manager, you:
1. **Receive trading requests** from users (stock symbols, analysis requests, portfolio evaluations)
2. **Coordinate the analyst team** to gather comprehensive market intelligence
3. **Facilitate researcher debates** between bullish and bearish perspectives
4. **Synthesize all insights** into actionable trading recommendations
5. **Evaluate risk** and make final approval/rejection decisions on trades

## Output Format

Present your final analysis in this structured format:

## 📊 Trading Analysis: [SYMBOL]

### Executive Summary
[One-paragraph overview of the recommendation]

### Analyst Findings
| Analyst | Signal | Key Insight |
|---------|--------|-------------|
| Fundamentals | 🟢/🟡/🔴 | [Brief finding] |
| Sentiment | 🟢/🟡/🔴 | [Brief finding] |
| News | 🟢/🟡/🔴 | [Brief finding] |
| Technical | 🟢/🟡/🔴 | [Brief finding] |

### Bull vs Bear Debate
**Bull Case**: [Key bullish arguments]
**Bear Case**: [Key bearish arguments]

### Risk Assessment
- Volatility Level: [High/Medium/Low]
- Liquidity: [Good/Fair/Poor]
- Portfolio Impact: [Analysis]

### 🎯 Final Recommendation
**Action**: [BUY/SELL/HOLD]
**Confidence**: [High/Medium/Low]
**Suggested Position**: [% of portfolio or share quantity]
**Target Price**: [If applicable]
**Stop Loss**: [Suggested level]

### ⚠️ Disclaimers
- This is AI-generated analysis for educational purposes only
- Not financial advice — always do your own research
- Past performance does not guarantee future results"""


def _print_step(step: str, detail: str = ""):
    """Print a formatted progress step."""
    print(f"\n{'─' * 60}")
    print(f"  {step}")
    if detail:
        print(f"  {detail}")
    print(f"{'─' * 60}")


def run_analyst_parallel(client: anthropic.Anthropic, ticker: str) -> dict:
    """Run all four analysts in parallel using a thread pool."""
    _print_step("STEP 1: ANALYST TEAM", f"Deploying 4 analysts for {ticker.upper()} in parallel...")

    analyst_fns = {
        "Fundamentals Analyst": run_fundamentals_analyst,
        "Sentiment Analyst": run_sentiment_analyst,
        "News Analyst": run_news_analyst,
        "Technical Analyst": run_technical_analyst,
    }

    reports = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(fn, client, ticker): name
            for name, fn in analyst_fns.items()
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                report = future.result()
                reports[name] = report
                print(f"  ✓ {name} completed")
            except Exception as e:
                reports[name] = f"Error: {e}"
                print(f"  ✗ {name} failed: {e}")

    return reports


def run_researcher_debate(
    client: anthropic.Anthropic,
    ticker: str,
    analyst_reports: dict,
    debate_rounds: int = 2,
) -> tuple[str, str]:
    """
    Run bullish/bearish researcher debate for the specified number of rounds.
    Returns (final_bull_case, final_bear_case).
    """
    _print_step(
        "STEP 2: RESEARCHER DEBATE",
        f"Running {debate_rounds} round(s) of bull vs bear debate..."
    )

    bull_argument = ""
    bear_argument = ""

    for round_num in range(1, debate_rounds + 1):
        print(f"\n  Round {round_num}/{debate_rounds}:")

        print(f"    → Bullish Researcher building case...", end="", flush=True)
        bull_argument = run_bullish_researcher(
            client, ticker, analyst_reports, bear_argument, round_num
        )
        print(" ✓")

        print(f"    → Bearish Researcher countering...", end="", flush=True)
        bear_argument = run_bearish_researcher(
            client, ticker, analyst_reports, bull_argument, round_num
        )
        print(" ✓")

    return bull_argument, bear_argument


def run_risk_assessment(
    client: anthropic.Anthropic,
    ticker: str,
    analyst_reports: dict,
    bull_case: str,
    bear_case: str,
) -> str:
    """Run risk manager assessment."""
    _print_step("STEP 3: RISK ASSESSMENT", "Risk Manager evaluating position risk...")

    report = run_risk_manager(client, ticker, analyst_reports, bull_case, bear_case)
    print("  ✓ Risk assessment complete")
    return report


def analyze_ticker(
    ticker: str,
    debate_rounds: int = 2,
    api_key: str | None = None,
) -> str:
    """
    Full multi-agent trading analysis workflow.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL", "TSLA")
        debate_rounds: Number of bull/bear debate rounds (1-3 recommended)
        api_key: Optional API key override (uses ANTHROPIC_API_KEY env var by default)

    Returns:
        The Portfolio Manager's final recommendation string.
    """
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    ticker = ticker.upper().strip()
    start_time = time.time()

    print(f"\n{'═' * 60}")
    print(f"  TRADING AGENTS — Analysis: {ticker}")
    print(f"{'═' * 60}")

    # Step 1: Parallel analyst team
    analyst_reports = run_analyst_parallel(client, ticker)

    # Step 2: Researcher debate
    bull_case, bear_case = run_researcher_debate(
        client, ticker, analyst_reports, debate_rounds=debate_rounds
    )

    # Step 3: Risk assessment
    risk_report = run_risk_assessment(
        client, ticker, analyst_reports, bull_case, bear_case
    )

    # Step 4: Portfolio Manager final synthesis (streamed)
    _print_step("STEP 4: PORTFOLIO MANAGER", "Synthesizing final recommendation (streaming)...")
    final_recommendation = run_portfolio_manager(
        client=client,
        ticker=ticker,
        analyst_reports=analyst_reports,
        bull_case=bull_case,
        bear_case=bear_case,
        risk_report=risk_report,
        system_prompt=PORTFOLIO_MANAGER_SYSTEM,
    )

    elapsed = time.time() - start_time
    print(f"\n{'═' * 60}")
    print(f"  Analysis complete in {elapsed:.1f}s")
    print(f"{'═' * 60}\n")

    return final_recommendation
