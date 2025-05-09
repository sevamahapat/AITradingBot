import pandas as pd
from dateutil import parser

def parse_date_safe(val):
    try:
        if pd.notnull(val):
            return parser.parse(val, dayfirst=False)  # handles both DD-MM-YYYY and MM/DD/YYYY
    except Exception:
        return pd.NaT
    return pd.NaT

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
    df["Amount_clean"] = pd.to_numeric(df["Amount_clean"], errors="coerce")

    for col in ["Activity Date", "Process Date", "Settle Date"]:
        df[col] = df[col].apply(parse_date_safe)
        # df[col] = pd.to_datetime(df[col], errors="coerce")

    df.to_csv("Cleaned_Trades_sample.csv", index=False)

    return df
