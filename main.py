#!/usr/bin/env python3
"""
TradingAgents — Multi-Agent Financial Trading Framework
Powered by Claude claude-opus-4-6 + yfinance

Usage:
    python main.py AAPL
    python main.py TSLA --rounds 3
    python main.py SPY NVDA MSFT          # Analyze multiple tickers sequentially

Environment:
    ANTHROPIC_API_KEY — required (or use .env file)
"""

import argparse
import os
import sys

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(
        description="TradingAgents — Multi-Agent Financial Trading Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py AAPL
  python main.py TSLA --rounds 3
  python main.py NVDA MSFT GOOGL
        """,
    )
    parser.add_argument(
        "tickers",
        nargs="+",
        help="One or more stock ticker symbols to analyze (e.g. AAPL TSLA SPY)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Number of bull/bear debate rounds (default: 2)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key (overrides ANTHROPIC_API_KEY env var)",
    )
    args = parser.parse_args()

    # Validate API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "Error: ANTHROPIC_API_KEY not set.\n"
            "Set it via environment variable or use --api-key flag.\n"
            "Get your key at: https://console.anthropic.com/",
            file=sys.stderr,
        )
        sys.exit(1)

    # Import here so env is loaded first
    from trading_agents.orchestrator import analyze_ticker

    for ticker in args.tickers:
        try:
            analyze_ticker(
                ticker=ticker,
                debate_rounds=args.rounds,
                api_key=api_key,
            )
        except KeyboardInterrupt:
            print("\n\nAnalysis interrupted by user.")
            sys.exit(0)
        except Exception as e:
            print(f"\nError analyzing {ticker}: {e}", file=sys.stderr)
            if len(args.tickers) == 1:
                sys.exit(1)


if __name__ == "__main__":
    main()
