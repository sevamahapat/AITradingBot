import pandas as pd

def clean_amount(val):
    if isinstance(val, str):
        val = val.strip()
        if "(" in val:
            return -float(val.replace("$", "").replace(",", "").replace("(", "").replace(")", ""))
        return float(val.replace("$", "").replace(",", ""))
    return None

def load_trading_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    df["Amount_clean"] = df["Amount"].apply(clean_amount)

    for col in ["Activity Date", "Process Date", "Settle Date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df.to_csv("Cleaned_Trades_sample.csv", index=False)

    return df
