import pandas as pd

def parse_order_date_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()

    # First try day-first parsing, which matches Zomato-style Order_Date better
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)

    # Fallback: try default parsing for already ISO-like values
    fallback_mask = parsed.isna()
    if fallback_mask.any():
        parsed.loc[fallback_mask] = pd.to_datetime(
            text.loc[fallback_mask], errors="coerce"
        )

    if parsed.notna().any():
        return parsed.dt.normalize()
    return parsed