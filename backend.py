"""
TradingAgents — FastAPI Data Backend

Serves yfinance stock data as JSON endpoints.
All Claude AI calls happen in the browser via Puter.js — no API key required.

Run with:
    uvicorn backend:app --host 0.0.0.0 --port 8000
"""

import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from trading_agents.tools import (
    get_company_info,
    get_financials,
    get_balance_sheet,
    get_technical_indicators,
    get_price_history,
    get_news,
    get_analyst_ratings,
    get_market_sentiment,
)

app = FastAPI(title="TradingAgents Data API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/stock/{ticker}")
def get_stock_data(ticker: str):
    """
    Fetch all market data for a ticker in one request.
    Returns company info, financials, technicals, news, and sentiment.
    """
    ticker = ticker.upper().strip()
    if not ticker.replace(".", "").isalpha() or len(ticker) > 6:
        raise HTTPException(status_code=400, detail="Invalid ticker symbol")

    return {
        "ticker": ticker,
        "company_info":   json.loads(get_company_info(ticker)),
        "financials":     json.loads(get_financials(ticker)),
        "balance_sheet":  json.loads(get_balance_sheet(ticker)),
        "technical":      json.loads(get_technical_indicators(ticker)),
        "price_history":  json.loads(get_price_history(ticker, "3mo")),
        "news":           json.loads(get_news(ticker)),
        "analyst_ratings":json.loads(get_analyst_ratings(ticker)),
        "sentiment":      json.loads(get_market_sentiment(ticker)),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# Serve the HTML frontend at root (must be last)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
