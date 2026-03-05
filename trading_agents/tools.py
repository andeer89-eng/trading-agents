"""
Market data tools powered by yfinance.
Each function returns a JSON string for use as a Claude tool result.
"""

import json
import yfinance as yf
import pandas as pd
from datetime import datetime


def _safe_json(obj) -> str:
    """Serialize to JSON, converting non-serializable types."""
    def default(o):
        if isinstance(o, float) and (o != o):  # NaN check
            return None
        try:
            return str(o)
        except Exception:
            return None
    return json.dumps(obj, default=default, indent=2)


def get_company_info(ticker: str) -> str:
    """Get company overview, sector, market cap, and description."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        result = {
            "name": info.get("longName"),
            "ticker": ticker.upper(),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "exchange": info.get("exchange"),
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "employees": info.get("fullTimeEmployees"),
            "description": (info.get("longBusinessSummary") or "")[:600],
            "website": info.get("website"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "beta": info.get("beta"),
        }
        return _safe_json(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_price_history(ticker: str, period: str = "3mo") -> str:
    """Get historical OHLCV price data and key statistics.
    period options: 1mo, 3mo, 6mo, 1y, 2y, 5y
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return json.dumps({"error": "No price data available"})

        close = hist["Close"]
        current = float(close.iloc[-1])
        start = float(close.iloc[0])

        # Recent closing prices (last 15 days)
        recent = {
            str(k.date()): round(float(v), 2)
            for k, v in close.tail(15).items()
        }

        result = {
            "period": period,
            "current_price": round(current, 2),
            "period_start_price": round(start, 2),
            "period_return_pct": round((current / start - 1) * 100, 2),
            "period_high": round(float(hist["High"].max()), 2),
            "period_low": round(float(hist["Low"].min()), 2),
            "avg_daily_volume": int(hist["Volume"].mean()),
            "last_volume": int(hist["Volume"].iloc[-1]),
            "recent_closes": recent,
        }
        return _safe_json(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_financials(ticker: str) -> str:
    """Get key financial ratios, income statement metrics, and valuation multiples."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        result = {
            "valuation": {
                "pe_ratio_trailing": info.get("trailingPE"),
                "pe_ratio_forward": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
                "ev_to_revenue": info.get("enterpriseToRevenue"),
            },
            "income": {
                "total_revenue": info.get("totalRevenue"),
                "revenue_growth_yoy": info.get("revenueGrowth"),
                "gross_margin": info.get("grossMargins"),
                "operating_margin": info.get("operatingMargins"),
                "profit_margin": info.get("profitMargins"),
                "ebitda": info.get("ebitda"),
                "net_income": info.get("netIncomeToCommon"),
                "eps_trailing": info.get("trailingEps"),
                "eps_forward": info.get("forwardEps"),
                "earnings_growth_yoy": info.get("earningsGrowth"),
                "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth"),
            },
            "dividends": {
                "dividend_yield": info.get("dividendYield"),
                "payout_ratio": info.get("payoutRatio"),
                "dividend_rate": info.get("dividendRate"),
            },
        }
        return _safe_json(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_balance_sheet(ticker: str) -> str:
    """Get balance sheet health: assets, debt, cash, and liquidity ratios."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        result = {
            "assets": {
                "total_assets": info.get("totalAssets"),
                "cash_and_equivalents": info.get("totalCash"),
                "cash_per_share": info.get("totalCashPerShare"),
            },
            "liabilities": {
                "total_debt": info.get("totalDebt"),
                "long_term_debt": info.get("longTermDebt"),
                "debt_to_equity": info.get("debtToEquity"),
            },
            "equity": {
                "book_value_per_share": info.get("bookValue"),
                "return_on_equity": info.get("returnOnEquity"),
                "return_on_assets": info.get("returnOnAssets"),
            },
            "liquidity": {
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio"),
                "free_cashflow": info.get("freeCashflow"),
                "operating_cashflow": info.get("operatingCashflow"),
            },
        }
        return _safe_json(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_technical_indicators(ticker: str) -> str:
    """Calculate RSI, MACD, Bollinger Bands, and moving averages."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        if hist.empty:
            return json.dumps({"error": "No data available"})

        close = hist["Close"]
        volume = hist["Volume"]
        current_price = float(close.iloc[-1])

        # RSI (14-period)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = (100 - (100 / (1 + gain / loss))).iloc[-1]

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line

        # Moving averages
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]

        # Bollinger Bands (20-period, 2 std)
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = (bb_mid + 2 * bb_std).iloc[-1]
        bb_lower = (bb_mid - 2 * bb_std).iloc[-1]

        # Volume trend
        avg_vol_20 = float(volume.rolling(20).mean().iloc[-1])
        vol_ratio = float(volume.iloc[-1]) / avg_vol_20 if avg_vol_20 > 0 else None

        def pct_vs(price, ma):
            if pd.isna(ma) or ma == 0:
                return None
            return round((price / ma - 1) * 100, 2)

        result = {
            "price": round(current_price, 2),
            "momentum": {
                "rsi_14": round(float(rsi), 2) if not pd.isna(rsi) else None,
                "rsi_signal": (
                    "oversold (<30)" if not pd.isna(rsi) and rsi < 30
                    else "overbought (>70)" if not pd.isna(rsi) and rsi > 70
                    else "neutral"
                ),
                "macd_line": round(float(macd_line.iloc[-1]), 4) if not pd.isna(macd_line.iloc[-1]) else None,
                "macd_signal": round(float(signal_line.iloc[-1]), 4) if not pd.isna(signal_line.iloc[-1]) else None,
                "macd_histogram": round(float(macd_hist.iloc[-1]), 4) if not pd.isna(macd_hist.iloc[-1]) else None,
                "macd_cross": "bullish" if not pd.isna(macd_hist.iloc[-1]) and float(macd_hist.iloc[-1]) > 0 else "bearish",
            },
            "moving_averages": {
                "ma_20": round(float(ma20), 2) if not pd.isna(ma20) else None,
                "ma_50": round(float(ma50), 2) if not pd.isna(ma50) else None,
                "ma_200": round(float(ma200), 2) if not pd.isna(ma200) else None,
                "vs_ma20_pct": pct_vs(current_price, ma20),
                "vs_ma50_pct": pct_vs(current_price, ma50),
                "vs_ma200_pct": pct_vs(current_price, ma200),
                "trend": (
                    "strong uptrend" if not pd.isna(ma50) and not pd.isna(ma200) and current_price > float(ma50) > float(ma200)
                    else "strong downtrend" if not pd.isna(ma50) and not pd.isna(ma200) and current_price < float(ma50) < float(ma200)
                    else "mixed"
                ),
            },
            "bollinger_bands": {
                "upper": round(float(bb_upper), 2) if not pd.isna(bb_upper) else None,
                "lower": round(float(bb_lower), 2) if not pd.isna(bb_lower) else None,
                "position": (
                    "near upper band" if not pd.isna(bb_upper) and current_price > float(bb_upper) * 0.97
                    else "near lower band" if not pd.isna(bb_lower) and current_price < float(bb_lower) * 1.03
                    else "within bands"
                ),
            },
            "volume": {
                "last_volume": int(volume.iloc[-1]),
                "avg_volume_20d": int(avg_vol_20),
                "volume_vs_avg_ratio": round(vol_ratio, 2) if vol_ratio else None,
            },
        }
        return _safe_json(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_news(ticker: str) -> str:
    """Get recent news headlines and summaries for a stock."""
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news or []
        formatted = []
        for item in news_items[:12]:
            ts = item.get("providerPublishTime", 0)
            published = datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else "Unknown"
            formatted.append({
                "title": item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "published": published,
                "summary": (item.get("summary") or "")[:400],
                "url": item.get("link", ""),
            })
        return _safe_json({"count": len(formatted), "articles": formatted})
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_analyst_ratings(ticker: str) -> str:
    """Get Wall Street analyst price targets and recommendation consensus."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        result = {
            "consensus": info.get("recommendationKey", "N/A"),
            "mean_rating": info.get("recommendationMean"),
            "target_price_mean": info.get("targetMeanPrice"),
            "target_price_high": info.get("targetHighPrice"),
            "target_price_low": info.get("targetLowPrice"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "num_analysts": info.get("numberOfAnalystOpinions"),
        }
        # Upside/downside to target
        current = result["current_price"]
        target = result["target_price_mean"]
        if current and target:
            result["upside_to_target_pct"] = round((target / current - 1) * 100, 2)

        # Recent ratings history
        try:
            recs = stock.recommendations
            if recs is not None and not recs.empty:
                recent = recs.tail(8).reset_index()
                result["recent_ratings"] = recent.to_dict(orient="records")
        except Exception:
            pass

        return _safe_json(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_market_sentiment(ticker: str) -> str:
    """Get short interest, institutional ownership, and insider activity."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        result = {
            "short_interest": {
                "short_ratio": info.get("shortRatio"),
                "short_percent_of_float": info.get("shortPercentOfFloat"),
            },
            "ownership": {
                "institutional_pct": info.get("heldPercentInstitutions"),
                "insider_pct": info.get("heldPercentInsiders"),
            },
            "options_signals": {
                "implied_volatility_pct": None,  # Not directly in yfinance info
                "beta": info.get("beta"),
                "52w_change": info.get("52WeekChange"),
                "sp500_52w_change": info.get("SandP52WeekChange"),
            },
            "float_shares": info.get("floatShares"),
            "shares_outstanding": info.get("sharesOutstanding"),
        }
        return _safe_json(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ─── Tool registry ────────────────────────────────────────────────────────────

TOOL_FUNCTIONS = {
    "get_company_info": get_company_info,
    "get_price_history": get_price_history,
    "get_financials": get_financials,
    "get_balance_sheet": get_balance_sheet,
    "get_technical_indicators": get_technical_indicators,
    "get_news": get_news,
    "get_analyst_ratings": get_analyst_ratings,
    "get_market_sentiment": get_market_sentiment,
}

# ─── Claude tool schemas ──────────────────────────────────────────────────────

TICKER_SCHEMA = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL, TSLA, SPY"}
    },
    "required": ["ticker"],
}

ALL_TOOLS = [
    {
        "name": "get_company_info",
        "description": "Get company overview: name, sector, industry, market cap, description, current price, 52-week range, and beta.",
        "input_schema": TICKER_SCHEMA,
    },
    {
        "name": "get_price_history",
        "description": "Get historical price data including period return, high/low, volume, and recent closing prices. Use period: 1mo, 3mo, 6mo, 1y, 2y, 5y.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
                "period": {
                    "type": "string",
                    "description": "Time period: 1mo, 3mo, 6mo, 1y, 2y, 5y",
                    "enum": ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_financials",
        "description": "Get financial ratios and metrics: P/E, PEG, P/B, revenue growth, margins, EPS, earnings growth, and dividend yield.",
        "input_schema": TICKER_SCHEMA,
    },
    {
        "name": "get_balance_sheet",
        "description": "Get balance sheet data: total assets, debt, cash, current ratio, quick ratio, return on equity, and free cash flow.",
        "input_schema": TICKER_SCHEMA,
    },
    {
        "name": "get_technical_indicators",
        "description": "Calculate technical indicators: RSI(14), MACD, Bollinger Bands, moving averages (20/50/200-day), and volume analysis.",
        "input_schema": TICKER_SCHEMA,
    },
    {
        "name": "get_news",
        "description": "Get the latest news headlines and summaries related to a stock from financial news sources.",
        "input_schema": TICKER_SCHEMA,
    },
    {
        "name": "get_analyst_ratings",
        "description": "Get Wall Street analyst consensus rating, price targets (mean/high/low), upside to target, and recent rating changes.",
        "input_schema": TICKER_SCHEMA,
    },
    {
        "name": "get_market_sentiment",
        "description": "Get market sentiment indicators: short interest, short ratio, institutional ownership, insider ownership, and beta.",
        "input_schema": TICKER_SCHEMA,
    },
]

# Subsets for each analyst role
FUNDAMENTALS_TOOLS = [t for t in ALL_TOOLS if t["name"] in {
    "get_company_info", "get_financials", "get_balance_sheet", "get_analyst_ratings"
}]

SENTIMENT_TOOLS = [t for t in ALL_TOOLS if t["name"] in {
    "get_market_sentiment", "get_news", "get_analyst_ratings"
}]

NEWS_TOOLS = [t for t in ALL_TOOLS if t["name"] in {
    "get_news", "get_company_info"
}]

TECHNICAL_TOOLS = [t for t in ALL_TOOLS if t["name"] in {
    "get_technical_indicators", "get_price_history"
}]


def execute_tool(name: str, tool_input: dict) -> str:
    """Dispatch a tool call to the appropriate function."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    return fn(**tool_input)
