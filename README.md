# AI Trading Risk Insight Chatbot

This project delivers a **proactive AI assistant** that analyzes trading behavior, calculates financial metrics, detects risk patterns, and gives intuitive, personalized advice to a trader, based on historical trade data (CSV).

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone [https://github.com/sevamahapat/ai-trading-bot.git](https://github.com/sevamahapat/AITradingBot.git)
cd ai-trading-bot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Environment Variables

Create a `.env` file with your Gemini API key: `GOOGLE_API_KEY=your_gemini_api_key`

#### How to Get Your Google Gemini API Key

1. **Go to the Google AI Studio**  
   Visit: [https://makersuite.google.com/app](https://makersuite.google.com/app)

2. **Sign in with your Google account**

3. **Click on "GET API Key" button**.

4. **Click "Create API Key" on top-right corner**  
   Follow the prompt to generate your key.

5. **Copy the key** and paste it into your project’s `.env` file like this: `GOOGLE_API_KEY=your_gemini_api_key`

### 5. Place Your Trade CSV

Add your trading file (e.g., Trades_sample.csv) into the project root.

## Run the Chatbot

```bash
python main.py
```

The chatbot will:

1. Load your trading data
2. Clean and preprocess it
3. Analyze deposits, top trades, expiry losses
4. Compute advanced risk advice
5. Perform behavioral analysis via LLM
6. Print a full Trader Summary Report

**Note:**

> If you'd prefer an **interactive chatbot experience** — where you ask questions one by one instead of receiving a full summary report —  
> **go to `main.py`** and:
>
> - **Comment out line 12** (the `generate_full_report()` function call)
> - **Uncomment lines 13 to 19**, which include the chatbot's input loop.
>
> This lets you ask the AI chatbot questions like:
>
> - “How much was deposited?”
> - “What advice would you give?”
> - “What was my most profitable trade?”

## Project Approach

### Step-by-Step Breakdown:

| Step            | Description                                                       | File                                 |
| --------------- | ----------------------------------------------------------------- | ------------------------------------ |
| Data Cleaning   | Normalize amounts, parse dates, handle missing values             | `data_loader.py`                     |
| Tool Setup      | Deposit tracking, profitable trades, expiry analysis, risk advice | `tools.py`                           |
| Risk Analysis   | Position sizing, volatility, drawdowns, revenge trading detection | `get_risk_advice()` in `tools.py`    |
| Expiry Matching | Links `OEXP` rows with `BTO` to compute real losses               | `calculate_expiry_loss_percentage()` |
| Forecasting     | Uses Prophet to predict account balance trends                    | Inside `get_risk_advice()`           |
| Final Synthesis | LLM evaluates all results and gives deep insights                 | `deep_insight()` in `tools.py`       |
| Agent Control   | Loads tools and models, runs queries                              | `agent.py`                           |
| Report          | Consolidates all observations and advice                          | `main.py` (`generate_full_report`)   |

## Logic & Tools Overview

- The `get_total_deposits` tool filters out rows where the transaction code is `ACH` and sums the deposit amounts to show how much money was added to the account.

- The `get_most_profitable_trade` tool looks at all non-deposit trades and identifies the one with the highest positive return (`Amount_clean`).

- The `calculate_expiry_loss_percentage` tool detects all options that expired worthless by matching expiration rows (`OEXP`) with their corresponding `BTO` entries, then computes the percentage of total losses that came from these.

- The `get_risk_advice` tool is the heart of the system. It runs several layered checks:

  - Calculates win/loss ratio to measure trading performance consistency.
  - Compares average gains vs. losses to detect whether the trader risks more than they earn.
  - Analyzes position sizing to flag trades that used more than 10% of the account value.
  - Calculates the standard deviation of gains to spot high variability in trade outcomes.
  - Detects emotional behavior such as revenge trading (e.g., increasing trade size after a loss).
  - Computes max drawdown by comparing peak and lowest account balance.
  - Uses the Prophet model to forecast account balance over the next 7 days and adjust risk advice accordingly.
  - Calculates reward-to-risk ratio using paired BTO and STC entries to evaluate trade quality.

- The `deep_insight` tool uses Gemini-1.5 Flash to synthesize outputs from all previous tools and generate higher-level behavioral observations and actionable psychological advice. It helps the user uncover patterns not easily seen from raw numbers alone.

- The `generate_full_report()` method in `main.py` strings together the results from all tools and outputs a complete trader performance summary in a clear, structured format.

This architecture blends rule-based analysis with LLM-powered reasoning, creating a chatbot that is both data-driven and intuitive.

## Key Strengths

- Tool-based agent orchestration (LangChain)
- LLM-enhanced reasoning with Gemini 1.5 Flash
- Custom financial logic, not generic prompts
- Combines factual analytics + behavioral coaching

## Future Ideas

- Streamlit frontend with upload + summary dashboard
- Visualizations like drawdown charts
- Better forecasting results and more accurate risk advice with a larger dataset
