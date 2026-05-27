import pandas as pd


def compute_normalized_delay_series(df: pd.DataFrame) -> pd.Series:
    if "observed_eta_min" not in df.columns:
        return pd.Series(0.0, index=df.index)

    values = pd.to_numeric(df["observed_eta_min"], errors="coerce")

    mean_val = float(values.mean()) if values.notna().any() else 0.0
    std_val = float(values.std(ddof=0)) if values.notna().any() else 0.0

    if std_val <= 1e-9:
        return pd.Series(0.0, index=df.index)

    return ((values - mean_val) / std_val).fillna(0.0)


def compute_customer_workload_contribution(
    travel_cost_min: float,
    service_time_min: float,
    normalized_delay: float,
    delay_lambda: float = 0.5,
) -> float:
    return (
        float(travel_cost_min)
        + float(service_time_min)
        + (float(delay_lambda) * float(normalized_delay))
    )


def compute_route_workload(
    stop_contributions: list[float],
) -> float:
    return float(sum(stop_contributions))