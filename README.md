# 📈 TradingAgents

**Multi-Agent Financial Trading Framework** powered by [Claude claude-opus-4-6](https://anthropic.com) + [yfinance](https://github.com/ranaroussi/yfinance).

A team of 8 specialized AI agents collaborates to produce comprehensive stock analysis — mirroring how a real trading firm operates.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Portfolio Manager                   │
│          (Orchestrator & Final Decision)             │
└──────────┬──────────────────────────────┬───────────┘
           │                              │
    ┌──────▼──────┐                ┌──────▼──────┐
    │  Researcher  │                │  Risk        │
    │  Debate      │                │  Manager     │
    │  Bull ↔ Bear │                │              │
    └──────┬──────┘                └─────────────┘
           │
    ┌──────▼──────────────────────────────────────┐
    │              Analyst Team (parallel)          │
    │  💰 Fundamentals  🧠 Sentiment               │
    │  📰 News          📉 Technical               │
    └──────────────────────────────────────────────┘
                         │
    ┌──────▼──────────────────────────────────────┐
    │           yfinance Market Data               │
    │  Prices · Financials · News · Indicators     │
    └─────────────────────────────────────────────┘
```

## 🤖 Agent Team

| Agent | Role | Data Sources |
|-------|------|-------------|
| **Fundamentals Analyst** | P/E, margins, growth, balance sheet | yfinance financials |
| **Sentiment Analyst** | Short interest, institutional ownership, crowd mood | yfinance info |
| **News Analyst** | Recent news, catalysts, macro impact | yfinance news feed |
| **Technical Analyst** | RSI, MACD, Bollinger Bands, moving averages | yfinance price history |
| **Bullish Researcher** | Builds bull case, counters bear arguments | Analyst reports |
| **Bearish Researcher** | Builds bear case, counters bull arguments | Analyst reports |
| **Risk Manager** | Volatility, position sizing, stop-loss | All reports |
| **Portfolio Manager** | Final BUY/SELL/HOLD recommendation | Everything above |

---

## 🚀 Quick Start (Local)

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/trading-agents.git
cd trading-agents
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set your API key

**Option A — `.env` file (recommended for local dev):**
```bash
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > .env
```

**Option B — Streamlit secrets:**
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and add your key
```

### 4. Run the web app
```bash
streamlit run app.py
```

### 5. Run the CLI
```bash
python main.py AAPL
python main.py TSLA --rounds 3
python main.py NVDA MSFT GOOGL
```

---

## 🌐 Deploy to Streamlit Community Cloud (Free)

### Step 1 — Push to GitHub
1. Create a new repo on [github.com/new](https://github.com/new)
2. Push your code:
```bash
git init
git add .
git commit -m "Initial commit: TradingAgents"
git remote add origin https://github.com/YOUR_USERNAME/trading-agents.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
2. Connect your GitHub account
3. Select your repository and set:
   - **Branch:** `main`
   - **Main file:** `app.py`
4. Click **Advanced settings** → **Secrets** and add:
```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```
5. Click **Deploy** — your app will be live in ~60 seconds!

---

## 📁 Project Structure

```
trading-agents/
├── app.py                        # Streamlit web interface
├── main.py                       # CLI entry point
├── requirements.txt
├── .gitignore
├── .streamlit/
│   ├── config.toml               # Dark theme + server settings
│   └── secrets.toml.example      # Template (copy & fill in locally)
└── trading_agents/
    ├── __init__.py
    ├── tools.py                  # yfinance market data tools
    ├── agents.py                 # 8 individual Claude agents
    └── orchestrator.py           # Multi-agent workflow coordination
```

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**.
It is **not financial advice**. Never trade based solely on AI analysis.
Always conduct your own research and consult a licensed financial advisor.

---

## 🙏 Built With

- [Anthropic Claude claude-opus-4-6](https://www.anthropic.com) — LLM backbone
- [yfinance](https://github.com/ranaroussi/yfinance) — Real-time market data
- [Streamlit](https://streamlit.io) — Web interface
