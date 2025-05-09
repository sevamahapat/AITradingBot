import pandas as pd
from langchain.tools import tool
from prophet import Prophet
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

df = None

def set_dataframe(dataframe):
    global df
    df = dataframe

@tool
def get_total_deposits() -> str:
    """Returns the total amount deposited into the account."""
    deposits = df[df["Trans Code"] == "ACH"]["Amount_clean"].sum()
    return f"Total deposits this month: ${deposits:,.2f}"

@tool
def get_most_profitable_trade() -> str:
    """Finds the most profitable trade."""
    trades = df[df["Trans Code"] != "ACH"]
    top = trades[trades["Amount_clean"] > 0].sort_values(by="Amount_clean", ascending=False).head(1)
    if not top.empty:
        desc = top.iloc[0]["Description"]
        amt = top.iloc[0]["Amount_clean"]
        return f"Most profitable trade: {desc}, Profit: ${amt:.2f}"
    return "No profitable trades found."

@tool
def calculate_expiry_loss_percentage() -> str:
    """Calculates what percentage of losses came from expired options."""
    expired = df[df["Trans Code"] == "OEXP"]
    bto_losses = 0
    for _, row in expired.iterrows():
        desc = row["Description"]
        match = df[
            (df["Description"].str.strip() == desc.replace("Option Expiration for ", "")) &
            (df["Trans Code"] == "BTO")
        ]
        bto_losses += match["Amount_clean"].sum()

    total_loss = df[df["Amount_clean"] < 0]["Amount_clean"].sum()
    if total_loss == 0:
        return "No losses recorded."
    
    loss_pct = abs(bto_losses / total_loss) * 100
    return f"{loss_pct:.2f}% of total losses came from options that expired worthless."

@tool
def get_risk_advice() -> str:
    """Provides risk management advice based on trade activity."""
    advice = []

    trades = df[df["Trans Code"] != "ACH"].copy()
    trades = trades[pd.to_numeric(trades["Amount_clean"], errors="coerce").notnull()]
    capital = df[df["Trans Code"] == "ACH"]["Amount_clean"].sum()
    
    # Win/Loss Ratio
    gains = trades[trades["Amount_clean"] > 0]["Amount_clean"]
    losses = trades[trades["Amount_clean"] < 0]["Amount_clean"]
    
    win_loss_ratio = len(gains) / len(losses) if len(losses) > 0 else float('inf')
    if win_loss_ratio < 1:
        advice.append(f"- Win/loss ratio is {win_loss_ratio:.2f}. Improve trade quality or timing.")
    
    # Average Gain vs. Loss
    avg_gain = gains.mean() if not gains.empty else 0
    avg_loss = abs(losses.mean()) if not losses.empty else 0
    if avg_loss > avg_gain:
        advice.append(f"- Average loss (${avg_loss:.2f}) exceeds average gain (${avg_gain:.2f}). Rethink your risk-reward strategy.")

    # Position Sizing Analysis
    if capital > 0:
        high_risk_trades = trades[abs(trades["Amount_clean"]) > 0.1 * capital]
        high_risk_pct = len(high_risk_trades) / len(trades)
        if high_risk_pct > 0:
            advice.append(f"- {high_risk_pct:.0%} of trades exceeded 10% of account value. Consider using smaller, consistent position sizes.")

    # Trade Variability
    if len(gains) >= 3:
        std_return = gains.std()
        if std_return > avg_gain:
            advice.append(f"- Your returns show high variability (${std_return:.2f} STD). Aim for more consistent outcomes.")
    
    # Expired Option Losses
    expired = df[df["Trans Code"] == "OEXP"]
    bto_losses = 0
    for _, row in expired.iterrows():
        desc = row["Description"]
        match = df[
            (df["Description"].str.strip() == desc.replace("Option Expiration for ", "")) &
            (df["Trans Code"] == "BTO")
        ]
        bto_losses += match["Amount_clean"].sum()
    
    total_loss = losses.sum()
    if total_loss:
        expired_pct = abs(bto_losses / total_loss) * 100
        if expired_pct > 20:
            advice.append(f"- {expired_pct:.2f}% of losses are due to expired options. Consider closing positions earlier.")

    # Risk-Reward Ratio
    rr_ratios = []
    grouped = trades.groupby("Description")
    for desc, group in grouped:
        buy = group[group["Trans Code"] == "BTO"]["Amount_clean"].sum()
        sell = group[group["Trans Code"] == "STC"]["Amount_clean"].sum()
        if buy < 0 and sell > 0:
            rr = sell / abs(buy)
            rr_ratios.append(rr)
    if rr_ratios:
        avg_rr = sum(rr_ratios) / len(rr_ratios)
        if avg_rr < 1.5:
            advice.append(f"- Average reward-to-risk ratio is {avg_rr:.2f}. Target trades with R:R ≥ 2.0.")

    # Behavioral Trend: Revenge Trading
    trades_sorted = trades.sort_values(by="Activity Date")
    revenge_flag = False
    for i in range(1, len(trades_sorted)):
        prev = trades_sorted.iloc[i - 1]
        curr = trades_sorted.iloc[i]
        if prev["Amount_clean"] < 0 and curr["Amount_clean"] < 0 and abs(curr["Amount_clean"]) > abs(prev["Amount_clean"]):
            revenge_flag = True
            break
    if revenge_flag:
        advice.append("- Possible revenge trading detected: Increasing losses after prior loss. Re-evaluate emotional discipline.")

    # Drawdown Analysis
    df_bal = df[["Activity Date", "Amount_clean"]].dropna()
    df_bal = df_bal.groupby("Activity Date").sum().cumsum().reset_index()
    df_bal.columns = ["ds", "y"]
    peak = df_bal["y"].cummax()
    drawdown = (df_bal["y"] - peak) / peak
    max_drawdown = drawdown.min()
    if max_drawdown < -0.2:
        advice.append(f"- Max drawdown was {abs(max_drawdown)*100:.2f}%. Consider pausing or downsizing during losing streaks.")

    # Forecasting
    try:
        df_forecast = df[["Activity Date", "Amount_clean"]].dropna()
        daily = df_forecast.groupby("Activity Date").sum().cumsum().reset_index()
        daily.columns = ["ds", "y"]

        model = Prophet()
        model.fit(daily)
        future = model.make_future_dataframe(periods=7)
        forecast = model.predict(future)
        trend = forecast["yhat"].diff().tail(7).mean()

        if trend < 0:
            advice.append("- Your forecasted balance is declining. Reduce risk exposure in upcoming trades.")
        else:
            advice.append("- Forecast trend looks stable. No immediate capital risk detected. Keep monitoring trade performance.")
    except Exception as e:
        advice.append("- Could not generate forecast: " + str(e))

    return "Risk Management Advice:\n" + "\n".join(advice)

@tool
def deep_insight(total_deposits: str, most_profitable_trade: str, expiry_loss: str, risk_advice: str) -> str:
    """
    Combines outputs from other tools and generates deeper, intuitive trading advice.
    Inputs: total deposits, most profitable trade, expiry loss %, and risk advice.
    Output: Key patterns, subtle insights, and behavioral suggestions.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

    prompt = f"""
    You are a senior trading coach AI. You've received this report from an automated analysis system:

    1. Total Deposits:\n{total_deposits}
    2. Most Profitable Trade:\n{most_profitable_trade}
    3. Expiry Loss %:\n{expiry_loss}
    4. Risk Advice:\n{risk_advice}

    Based on the above:
    - Identify any psychological or behavioral trading patterns.
    - Provide 2-3 intuitive, human-level insights that go beyond basic numbers.
    - Offer specific adjustments to trading strategy or mindset.
    - Keep it concise, motivational, and actionable.
    """

    response = llm.invoke(prompt)
    return response.content.strip()
