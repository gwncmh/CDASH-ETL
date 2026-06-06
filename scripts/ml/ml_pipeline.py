"""
ml_pipeline.py – Crime Hotspot Forecasting v24 groupblend confidence ranker
=================================================================
Airflow PythonOperator callable for BigData production.

Pipeline:
  BigQuery enriched_crime_data → daily H3 grid → v20 feature engineering
  → temporal/spatial/context LightGBM GroupBlend HOTSPOT ranker → top-k dashboard risk levels
  → prediction_results BigQuery.

Target:
  next_hotspot = 1 if next_total >= 2, else 0.

Dashboard risk policy:
  CRITICAL = top 5% H3 cells by hotspot_score
  HIGH     = next 10% cells, cumulative top 15%
  MEDIUM   = next 15% cells, cumulative top 30%
  LOW      = remaining 70%

This avoids the old failure mode where a fixed probability threshold could mark
an excessive share of the city as HIGH/CRITICAL.

Output schema keeps backward-compatible crime_probability and adds hotspot_score,
TopK metrics, GroupBlend scores, confidence fields and selected context fields.
"""

from __future__ import annotations

import logging
import os
import pickle
import warnings
from typing import Optional
import gc
import numpy as np
import pandas as pd
from google.cloud import bigquery

warnings.filterwarnings("ignore")
log = logging.getLogger("ML_Pipeline_v24_groupblend_confidence")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONSTANTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL_VERSION_PREFIX = "hotspot_v24_groupblend_confidence"

# Fixed GroupBlend_VAL_best from Kaggle Run 3.
GROUP_BLEND_WEIGHTS = {"temporal": 0.1, "spatial": 0.6, "context": 0.3}

# Rolling split windows (ngày)
TRAIN_DAYS       = 220
CALIB_DAYS       = 14
VAL_DAYS         = 7
TEST_DAYS        = 7
MIN_HISTORY_DAYS = 60

RANDOM_STATE = 42

# XGBoost hyperparameters from Kaggle v20-target-safe.
XGB_PARAMS = dict(
    n_estimators          = 1800,
    learning_rate         = 0.025,
    max_depth             = 4,
    min_child_weight      = 8,
    subsample             = 0.90,
    colsample_bytree      = 0.75,
    colsample_bylevel     = 0.85,
    colsample_bynode      = 0.90,
    reg_alpha             = 0.30,
    reg_lambda            = 4.0,
    objective             = "binary:logistic",
    eval_metric           = "aucpr",
    tree_method           = "hist",
    random_state          = RANDOM_STATE,
    n_jobs                = -1,
    early_stopping_rounds = 120,
)

THRESHOLD_GRID = np.arange(0.05, 0.65, 0.01).tolist()

# v20 experiment controls
RECENT_DAYS_TO_KEEP = 320
FLOAT_POLICY = "float32"

# Binary target: HOTSPOT means tomorrow count >= 2.
TARGET_COL = "next_hotspot"
CLASS_WEIGHT_POWER = 0.70
CLASS_WEIGHT_CAP = 12.0
CLASS_WEIGHT_EXTRA_BOOST = {1: 1.25}
EMERGING_POSITIVE_WEIGHT_BOOST = 1.25
LOW_MID_PRIOR_POSITIVE_WEIGHT_BOOST = 1.35
NEIGHBOR_SPIKE_POSITIVE_WEIGHT_BOOST = 1.15
MAX_SAMPLE_WEIGHT_MULTIPLIER = 1.75

# Threshold is diagnostic only. Dashboard risk uses top-k ranking.
USE_PRED_RATE_GUARD = True
MAX_PRED_RATE_MULTIPLIER = 2.20

DROP_REDUNDANT_PRIOR_FEATURES = True
DROP_STRONG_RAW_PRIORS = True
DROP_REDUNDANT_LONG_MEMORY_SUMS = True
DROP_DIRECT_STATIC_CONTEXT_FEATURES = True
DROP_AUDIT_ONLY_PRIOR_BAND_FEATURES = True
AUDIT_ONLY_PRIOR_BAND_FEATURES = {"prior_band_numeric", "is_low_mid_prior", "is_high_prior"}

DASHBOARD_TOP_CRITICAL = 0.05
DASHBOARD_TOP_HIGH = 0.15
DASHBOARD_TOP_MEDIUM = 0.30
DASHBOARD_EVAL_KS = [0.05, 0.10, 0.15, 0.25, 0.30]

# Ngày lễ Chicago ảnh hưởng đến tội phạm (Routine Activity Theory)
_HOLIDAYS_RAW = [
    "2022-01-01", "2022-01-17", "2022-05-30", "2022-07-04",
    "2022-09-05", "2022-11-24", "2022-11-25", "2022-12-24", "2022-12-25", "2022-12-31",
    "2023-01-01", "2023-01-16", "2023-02-20", "2023-05-29", "2023-07-04",
    "2023-09-04", "2023-10-09", "2023-11-23", "2023-11-24",
    "2023-12-24", "2023-12-25", "2023-12-31",
    "2024-01-01", "2024-01-15", "2024-05-27", "2024-07-04",
    "2024-09-02", "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-20", "2025-05-26", "2025-07-04",
    "2025-09-01", "2025-11-27", "2025-12-25",
]
ALL_HOLIDAYS: set = set(pd.to_datetime(_HOLIDAYS_RAW).tolist())

# Crime types classification
VIOLENT_TYPES  = {
    "BATTERY", "ASSAULT", "ROBBERY", "HOMICIDE",
    "CRIMINAL SEXUAL ASSAULT", "SEX OFFENSE", "KIDNAPPING", "INTIMIDATION",
}
PROPERTY_TYPES = {
    "THEFT", "BURGLARY", "MOTOR VEHICLE THEFT",
    "CRIMINAL DAMAGE", "DECEPTIVE PRACTICE", "ARSON",
}

# BigQuery schema cho prediction_results
BQ_PREDICTION_SCHEMA = [
    bigquery.SchemaField("prediction_date",   "DATE"),
    bigquery.SchemaField("h3_index",          "STRING"),
    bigquery.SchemaField("crime_probability", "FLOAT"),  # backward-compatible alias of hotspot_score
    bigquery.SchemaField("hotspot_score",     "FLOAT"),
    bigquery.SchemaField("risk_level",        "STRING"),
    bigquery.SchemaField("model_version",     "STRING"),
    bigquery.SchemaField("created_at",        "TIMESTAMP"),
    bigquery.SchemaField("pred_hotspot",      "INTEGER"),
    bigquery.SchemaField("risk_rank",         "INTEGER"),
    bigquery.SchemaField("risk_rank_pct",     "FLOAT"),
    bigquery.SchemaField("confidence_score", "FLOAT"),
    bigquery.SchemaField("confidence_level", "STRING"),
    bigquery.SchemaField("temporal_score", "FLOAT"),
    bigquery.SchemaField("spatial_score", "FLOAT"),
    bigquery.SchemaField("context_score", "FLOAT"),
    bigquery.SchemaField("group_score_std", "FLOAT"),
    bigquery.SchemaField("group_score_range", "FLOAT"),
    bigquery.SchemaField("model_agreement", "FLOAT"),
    bigquery.SchemaField("score_margin_to_cutoff", "FLOAT"),
    bigquery.SchemaField("rank_margin_confidence", "FLOAT"),
    bigquery.SchemaField("rank_extremity", "FLOAT"),
    bigquery.SchemaField("group_w_temporal", "FLOAT"),
    bigquery.SchemaField("group_w_spatial", "FLOAT"),
    bigquery.SchemaField("group_w_context", "FLOAT"),
    bigquery.SchemaField("val_auc",           "FLOAT"),
    bigquery.SchemaField("val_ap",            "FLOAT"),
    bigquery.SchemaField("val_f1",            "FLOAT"),
    bigquery.SchemaField("test_auc",          "FLOAT"),
    bigquery.SchemaField("test_ap",           "FLOAT"),
    bigquery.SchemaField("test_f1",           "FLOAT"),
    bigquery.SchemaField("val_top05_recall",  "FLOAT"),
    bigquery.SchemaField("val_top05_lift",    "FLOAT"),
    bigquery.SchemaField("val_top15_recall",  "FLOAT"),
    bigquery.SchemaField("val_top15_lift",    "FLOAT"),
    bigquery.SchemaField("val_top30_recall",  "FLOAT"),
    bigquery.SchemaField("val_top30_lift",    "FLOAT"),
    bigquery.SchemaField("test_top05_recall", "FLOAT"),
    bigquery.SchemaField("test_top05_lift",   "FLOAT"),
    bigquery.SchemaField("test_top15_recall", "FLOAT"),
    bigquery.SchemaField("test_top15_lift",   "FLOAT"),
    bigquery.SchemaField("test_top30_recall", "FLOAT"),
    bigquery.SchemaField("test_top30_lift",   "FLOAT"),
    # selected context / debug columns
    bigquery.SchemaField("community_area",    "FLOAT"),
    bigquery.SchemaField("hardship_index",    "FLOAT"),
    bigquery.SchemaField("unemployment_rate", "FLOAT"),
    bigquery.SchemaField("population_density","FLOAT"),
    bigquery.SchemaField("h3_train_percentile", "FLOAT"),
    bigquery.SchemaField("h3_train_active_rate", "FLOAT"),
    bigquery.SchemaField("roll_mean_7",       "FLOAT"),
    bigquery.SchemaField("roll_mean_30",      "FLOAT"),
    bigquery.SchemaField("zero_streak",       "FLOAT"),
    bigquery.SchemaField("nbr_r1_mean",       "FLOAT"),
    bigquery.SchemaField("nbr_r1_active_rate", "FLOAT"),
    bigquery.SchemaField("nbr_r1_active_count", "FLOAT"),
    bigquery.SchemaField("near_repeat_3d",    "FLOAT"),
    bigquery.SchemaField("arrest_rate_7",     "FLOAT"),
    bigquery.SchemaField("is_holiday",        "FLOAT"),
]

# ─────────────────────────────────────────────────────────────────────────────
# 1.  UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def safe_ratio(
    num: np.ndarray | pd.Series,
    den: np.ndarray | pd.Series,
    default: float = 0.0,
    clip: Optional[float] = None,
) -> np.ndarray:
    """RAM-safe ratio for tree models. float32 is enough for XGBoost hist."""
    dtype = np.float32 if FLOAT_POLICY == "float32" else np.float64
    num = np.asarray(num, dtype=dtype)
    den = np.asarray(den, dtype=dtype)
    out = np.full(num.shape, default, dtype=dtype)
    mask = np.abs(den) > dtype(1e-12)
    out[mask] = num[mask] / den[mask]
    if clip is not None:
        out = np.clip(out, -clip, clip).astype(dtype)
    return out.astype(dtype, copy=False)

def add_cyclical_date_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    d = pd.to_datetime(df[date_col])
    df["dayofweek"]  = d.dt.dayofweek
    df["month"]      = d.dt.month
    df["dayofyear"]  = d.dt.dayofyear
    df["is_weekend"] = d.dt.dayofweek.isin([5, 6]).astype(int)
    df["is_monday"]  = d.dt.dayofweek.eq(0).astype(int)
    df["is_friday"]  = d.dt.dayofweek.eq(4).astype(int)
    df["dow_sin"]    = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"]    = np.sin(2 * np.pi * df["dayofyear"] / 366)
    df["doy_cos"]    = np.cos(2 * np.pi * df["dayofyear"] / 366)
    return df


def holiday_features(dates_series: pd.Series, all_holidays: set) -> pd.DataFrame:
    """Tính is_holiday và khoảng cách đến ngày lễ gần nhất (Routine Activity Theory)."""
    holidays_arr = np.array(sorted(all_holidays), dtype="datetime64[D]")
    dates_np     = dates_series.values.astype("datetime64[D]")
    is_hol       = np.isin(dates_np, holidays_arr).astype(int)
    days_from, days_to = [], []
    for d in dates_np:
        diffs  = holidays_arr.astype(np.int64) - np.int64(d)
        past   = diffs[diffs <= 0]
        future = diffs[diffs > 0]
        days_from.append(int(abs(past.max()))   if len(past)   > 0 else 365)
        days_to.append(  int(abs(future.min())) if len(future) > 0 else 365)
    return pd.DataFrame(
        {
            "is_holiday":        is_hol,
            "days_from_holiday": np.clip(days_from, 0, 30),
            "days_to_holiday":   np.clip(days_to,   0, 30),
            "holiday_window":    (
                (np.array(days_from) <= 2) | (np.array(days_to) <= 2)
            ).astype(float),
        },
        index=dates_series.index,
    ).astype(float)


def sanitize_numeric(
    df: pd.DataFrame,
    cols: list[str],
    medians: Optional[pd.Series] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Select only feature columns to avoid copying the whole large dataframe."""
    x = df.loc[:, cols].copy()
    for c in cols:
        if c not in x.columns:
            x[c] = 0.0
        x[c] = pd.to_numeric(x[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = x.median(numeric_only=True).fillna(0.0)
    x = x.fillna(medians).fillna(0.0)
    dtype = np.float32 if FLOAT_POLICY == "float32" else np.float64
    return x.astype(dtype), medians.astype(dtype)

def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid: list[float] = THRESHOLD_GRID,
) -> tuple[float, float]:
    from sklearn.metrics import f1_score as _f1
    best_t, best_f1 = 0.5, 0.0
    for t in grid:
        pred = (y_prob >= t).astype(int)
        f    = _f1(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t, best_f1


def risk_level_from_prob(prob: pd.Series) -> pd.Series:
    """Backward-compatible fixed-bin helper; not used for v20 dashboard output."""
    return pd.cut(
        prob,
        bins=[0.0, 0.30, 0.60, 0.80, 1.01],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        right=False,
        include_lowest=True,
    ).astype(str)


def assign_dashboard_risk_levels_by_rank(score: pd.Series) -> pd.Series:
    """Assign exactly bounded dashboard groups on one prediction day.

    CRITICAL: top 5%; HIGH: next 10%; MEDIUM: next 15%; LOW: rest.
    This is the production guard against marking too much of the city as hot.
    """
    n = len(score)
    if n == 0:
        return pd.Series([], dtype="object")
    rank_pct = score.rank(method="first", ascending=False) / float(n)
    risk = pd.Series("LOW", index=score.index, dtype="object")
    risk.loc[rank_pct <= DASHBOARD_TOP_MEDIUM] = "MEDIUM"
    risk.loc[rank_pct <= DASHBOARD_TOP_HIGH] = "HIGH"
    risk.loc[rank_pct <= DASHBOARD_TOP_CRITICAL] = "CRITICAL"
    return risk


def precision_recall_lift_at_daily_k(
    eval_df: pd.DataFrame,
    score_col: str = "hotspot_score",
    actual_col: str = "actual_hotspot",
    k_frac: float = 0.05,
) -> dict:
    """Daily TopK precision/recall/lift for rare hotspot ranking."""
    rows = []
    for d, part in eval_df.groupby("date"):
        part = part.sort_values(score_col, ascending=False)
        n = len(part)
        if n == 0:
            continue
        k = max(1, int(np.ceil(k_frac * n)))
        top = part.head(k)
        total_pos = float(part[actual_col].sum())
        base_rate = total_pos / max(n, 1)
        hit = float(top[actual_col].sum())
        precision = hit / max(k, 1)
        recall = hit / max(total_pos, 1.0)
        lift = precision / max(base_rate, 1e-12)
        rows.append({"date": d, "precision": precision, "recall": recall, "lift": lift, "k": k, "base_rate": base_rate})
    if not rows:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "lift_at_k": 0.0}
    m = pd.DataFrame(rows)
    return {
        "precision_at_k": float(m["precision"].mean()),
        "recall_at_k": float(m["recall"].mean()),
        "lift_at_k": float(m["lift"].mean()),
    }


def evaluate_ranker(
    df_part: pd.DataFrame,
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> dict:
    """Combined threshold diagnostics + daily ranking metrics."""
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
    pred = (score >= threshold).astype(int)
    out = {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) > 1 else 0.5,
        "ap": float(average_precision_score(y_true, score)) if len(np.unique(y_true)) > 1 else float(np.mean(y_true)),
        "pred_rate": float(np.mean(pred)),
        "base_rate": float(np.mean(y_true)),
    }
    e = df_part[["date", "h3_index"]].copy()
    e["actual_hotspot"] = y_true.astype(int)
    e["hotspot_score"] = score.astype(float)
    for k in DASHBOARD_EVAL_KS:
        r = precision_recall_lift_at_daily_k(e, k_frac=k)
        tag = f"top{int(k*100):02d}"
        out[f"{tag}_precision"] = r["precision_at_k"]
        out[f"{tag}_recall"] = r["recall_at_k"]
        out[f"{tag}_lift"] = r["lift_at_k"]
    return out


def model_version_str() -> str:
    return pd.Timestamp.now().strftime("%Y-%m-%d") + f".{MODEL_VERSION_PREFIX}"


def feature_family_name(feature: str) -> str:
    """Group feature names for GroupBlend and interpretability."""
    f = str(feature)
    if f in AUDIT_ONLY_PRIOR_BAND_FEATURES:
        return "AUDIT_ONLY_PRIOR_BAND"
    if any(x in f for x in ["h3_train", "comm_train", "prior", "active_rate_vs_train"]):
        return "Spatial prior / prior-relative"
    if f.startswith("sim_") or "sim_cluster" in f or "cluster_train" in f or "h3_vs_sim" in f:
        return "Similarity graph"
    if "parent_r" in f or "h3_vs_parent" in f or "h3_share_parent" in f:
        return "Parent H3 multi-resolution"
    if "nbr_" in f or "_nbr_" in f or "contagion" in f:
        return "H3 neighbor spillover"
    if any(x in f for x in ["roll", "lag", "ewm", "burst", "shock", "spike", "streak", "trend", "same_dow", "recent", "activation", "decay"]):
        return "Temporal / momentum"
    if any(x in f for x in ["near_repeat", "reactivation", "arrest", "deterrence", "violent", "property", "theft", "battery", "narcotics", "crime_type_entropy", "dominant_type"]):
        return "Criminology behavior"
    if any(x in f for x in ["holiday", "weekend", "friday", "monday", "dow_", "month_", "doy_"]):
        return "Calendar / holiday"
    if any(x in f for x in ["temp", "precip", "wind"]):
        return "Weather"
    if any(x in f for x in ["dist_nearest", "near_school", "near_station", "near_park", "near_nightlife", "nightlife"]):
        return "POI interaction"
    if any(x in f for x in ["hardship", "unemployment", "population", "income", "density", "per_1000pop"]):
        return "Socioeconomic / density"
    if any(x in f for x in ["city_", "comm_", "h3_city_share", "h3_recent_vs_comm"]):
        return "City/community context"
    if any(x in f for x in ["night", "evening", "morning", "afternoon"]):
        return "Time-of-day profile"
    return "Other"


def feature_blend_bucket(feature: str) -> str:
    fam = feature_family_name(feature)
    if fam in {"Temporal / momentum", "Criminology behavior", "Calendar / holiday", "Time-of-day profile", "Weather"}:
        return "temporal"
    if fam in {"Spatial prior / prior-relative", "H3 neighbor spillover", "Similarity graph", "Parent H3 multi-resolution", "City/community context"}:
        return "spatial"
    if fam in {"Socioeconomic / density", "POI interaction"}:
        return "context"
    return "context"


def reduce_mem_usage(
    df: pd.DataFrame,
    exclude: tuple[str, ...] = ("h3_index", "date", "primary_type"),
) -> pd.DataFrame:
    """Downcast numeric columns to keep Airflow/Kaggle memory stable."""
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_float_dtype(df[c]):
            if FLOAT_POLICY == "float32":
                df[c] = pd.to_numeric(df[c], downcast="float")
            else:
                df[c] = df[c].astype("float64", copy=False)
        elif pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="integer")
    gc.collect()
    return df


def find_best_threshold_guarded(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid: list[float] = THRESHOLD_GRID,
    max_multiplier: float = MAX_PRED_RATE_MULTIPLIER,
) -> tuple[float, float, pd.DataFrame]:
    """Optimize F1 while penalizing thresholds that predict too many hotspots."""
    from sklearn.metrics import f1_score
    base_rate = float(np.mean(y_true))
    max_pred_rate = min(0.95, max_multiplier * base_rate)
    rows = []
    best_t, best_score, best_f1 = 0.5, -1.0, 0.0
    for t in grid:
        pred = (y_prob >= t).astype(int)
        pred_rate = float(pred.mean())
        f = f1_score(y_true, pred, zero_division=0)
        penalty = max(0.0, pred_rate - max_pred_rate) * 0.35
        score = f - penalty
        rows.append((t, f, pred_rate, score))
        if score > best_score:
            best_t, best_score, best_f1 = t, score, f
    return best_t, best_f1, pd.DataFrame(
        rows, columns=["threshold", "f1", "pred_rate", "guarded_score"]
    )


def h3_grid_disk_compat(h3lib, cell: str, k: int):
    """Compatible with h3 v3/v4."""
    if hasattr(h3lib, "grid_disk"):
        return h3lib.grid_disk(cell, k)
    if hasattr(h3lib, "k_ring"):
        return h3lib.k_ring(cell, k)
    raise AttributeError("h3 library has neither grid_disk nor k_ring")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOADING FROM BIGQUERY
# ─────────────────────────────────────────────────────────────────────────────

def load_enriched_data(
    client: bigquery.Client,
    project_id: str,
    dataset: str,
    enriched_table: str,
    lookback_days: int = RECENT_DAYS_TO_KEEP,
) -> pd.DataFrame:
    """
    Kéo enriched_crime_data từ BigQuery.
    Chỉ lấy ~lookback_days ngày gần nhất để tránh load toàn bộ lịch sử.
    """
    log.info("[ML] Loading enriched_crime_data from BigQuery (last %d days)...", lookback_days)

    query = f"""
        WITH max_src_date AS (
            SELECT MAX(DATE(date)) AS max_date
            FROM `{project_id}.{dataset}.{enriched_table}`
        )
        SELECT
            h3_index,
            DATE(date)                          AS date,
            primary_type,
            community_area,
            latitude,
            longitude,
            hour_of_day,
            day_of_week,
            month,
            -- weather
            temp_max,
            CAST(NULL AS FLOAT64)               AS temp_min,
            precipitation,
            wind_speed,
            -- socioeconomic
            unemployment_rate,
            hardship_index,
            population_density,
            per_capita_income,
            -- poi
            dist_nearest_school,
            dist_nearest_station,
            dist_nearest_park,
            dist_nearest_nightlife,
            -- crime rolling
            crime_density_7d,
            crime_density_30d,
            arrest_rate,
            -- arrest flag (derive from arrest_rate proxy; actual boolean not in gold)
            CAST(ROUND(arrest_rate) AS INT64)   AS arrest_flag,
            -- type flags
            CASE WHEN UPPER(TRIM(primary_type))
                      IN ('BATTERY','ASSAULT','ROBBERY','HOMICIDE',
                          'CRIMINAL SEXUAL ASSAULT','SEX OFFENSE',
                          'KIDNAPPING','INTIMIDATION')
                 THEN 1 ELSE 0 END              AS is_violent,
            CASE WHEN UPPER(TRIM(primary_type))
                      IN ('THEFT','BURGLARY','MOTOR VEHICLE THEFT',
                          'CRIMINAL DAMAGE','DECEPTIVE PRACTICE','ARSON')
                 THEN 1 ELSE 0 END              AS is_property,
            CASE WHEN UPPER(TRIM(primary_type)) = 'THEFT'
                 THEN 1 ELSE 0 END              AS is_theft,
            CASE WHEN UPPER(TRIM(primary_type)) = 'BATTERY'
                 THEN 1 ELSE 0 END              AS is_battery,
            CASE WHEN REGEXP_CONTAINS(UPPER(primary_type), r'NARCOTIC|DRUG')
                 THEN 1 ELSE 0 END              AS is_narcotics,
            CASE WHEN hour_of_day IN (0,1,2,3,4,5,22,23)
                 THEN 1 ELSE 0 END              AS is_night,
            CASE WHEN hour_of_day BETWEEN 18 AND 21
                 THEN 1 ELSE 0 END              AS is_evening,
            CASE WHEN hour_of_day BETWEEN 6 AND 11
                 THEN 1 ELSE 0 END              AS is_morning,
            CASE WHEN hour_of_day BETWEEN 12 AND 17
                 THEN 1 ELSE 0 END              AS is_afternoon
        FROM `{project_id}.{dataset}.{enriched_table}` AS src
        CROSS JOIN max_src_date
        WHERE DATE(src.date) >= DATE_SUB(max_src_date.max_date, INTERVAL {lookback_days} DAY)
          AND h3_index IS NOT NULL
          AND latitude  IS NOT NULL
          AND longitude IS NOT NULL
    """
    df = client.query(query).to_dataframe()
    if df.empty:
        raise ValueError(
            "No enriched crime rows were loaded from BigQuery. "
            "Check table name, date column, and lookback_days."
        )
    df["date"] = pd.to_datetime(df["date"]).dt.floor("D")
    log.info("[ML] Loaded %d rows, date range: %s → %s",
             len(df), df["date"].min().date(), df["date"].max().date())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  AGGREGATE: RAW EVENTS → DAILY PER H3
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_daily(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Tổng hợp dữ liệu crime event-level thành 1 dòng per (h3_index × date).
    Sau đó mở rộng Cartesian để mọi (H3, ngày) đều có dòng (kể cả 0 crime).
    """
    log.info("[ML] Aggregating daily per H3...")
    raw = df_raw.copy()

    mean_candidates = [
        "community_area", "temp_max", "temp_min", "precipitation", "wind_speed",
        "unemployment_rate", "hardship_index", "population_density",
        "per_capita_income", "dist_nearest_school", "dist_nearest_station",
        "dist_nearest_park", "dist_nearest_nightlife",
    ]
    mean_cols = [c for c in mean_candidates if c in raw.columns]
    for c in mean_cols:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    # Arrest: derive from arrest_flag or arrest_rate
    if "arrest_flag" in raw.columns:
        raw["arrest_num"] = raw["arrest_flag"].fillna(0).astype(int)
    elif "arrest_rate" in raw.columns:
        raw["arrest_num"] = (raw["arrest_rate"] > 0.5).astype(int)
    else:
        raw["arrest_num"] = 0

    # Assign case_number proxy nếu không có
    if "case_number" not in raw.columns:
        raw["case_number"] = np.arange(len(raw)).astype(str)

    # Preserve both type counts and time-of-day ratios.
    for c in ["is_morning", "is_afternoon"]:
        if c not in raw.columns:
            raw[c] = 0
    agg_spec: dict = {
        "case_number":  "count",
        "is_violent":   "sum",
        "is_property":  "sum",
        "is_theft":     "sum",
        "is_battery":   "sum",
        "is_narcotics": "sum",
        "arrest_num":   "sum",
        "is_night":     "mean",
        "is_evening":   "mean",
        "is_morning":   "mean",
        "is_afternoon": "mean",
    }
    for c in mean_cols:
        agg_spec[c] = "mean"

    daily = raw.groupby(["h3_index", "date"], as_index=False).agg(agg_spec)
    daily = daily.rename(columns={"case_number": "total_crimes"})

    # Cartesian expand: mọi H3 × ngày đều xuất hiện
    all_h3    = np.array(sorted(raw["h3_index"].unique()))
    all_dates = pd.date_range(raw["date"].min(), raw["date"].max(), freq="D")
    full_idx  = pd.MultiIndex.from_product(
        [all_h3, all_dates], names=["h3_index", "date"]
    )
    daily = daily.set_index(["h3_index", "date"]).reindex(full_idx).reset_index()

    count_cols = [
        "total_crimes", "is_violent", "is_property",
        "is_theft", "is_battery", "is_narcotics", "arrest_num",
    ]
    for c in count_cols:
        daily[c] = daily[c].fillna(0).astype(float)

    # Forward-fill context columns (thời tiết, socioeconomic không thay đổi nhiều)
    context_cols = [c for c in daily.columns if c not in ["h3_index", "date"] + count_cols]
    for c in context_cols:
        daily[c] = daily.groupby("h3_index")[c].transform(lambda s: s.ffill().bfill())
        if pd.api.types.is_numeric_dtype(daily[c]):
            daily[c] = daily[c].fillna(daily[c].median())

    daily = add_cyclical_date_features(daily)
    daily["crime_today"]         = (daily["total_crimes"] > 0).astype(float)
    daily["violent_ratio_day"]   = safe_ratio(daily["is_violent"],   daily["total_crimes"], clip=1)
    daily["property_ratio_day"]  = safe_ratio(daily["is_property"],  daily["total_crimes"], clip=1)
    daily["theft_ratio_day"]     = safe_ratio(daily["is_theft"],     daily["total_crimes"], clip=1)
    daily["narcotics_ratio_day"] = safe_ratio(daily["is_narcotics"], daily["total_crimes"], clip=1)

    daily = reduce_mem_usage(daily)
    log.info("[ML] Daily grid: %s rows, %d H3 cells, %d dates | RAM-safe downcast done",
             f"{len(daily):,}", len(all_h3), len(all_dates))
    return daily, all_h3


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SPATIAL SPILLOVER: H3 NEIGHBOR PRECOMPUTE
# ─────────────────────────────────────────────────────────────────────────────

def build_neighbor_maps(all_h3: np.ndarray) -> tuple[dict, dict, bool]:
    """
    Dùng thư viện h3 tính ring-1 và ring-2 neighbor của mỗi ô.
    Fallback gracefully nếu h3 chưa cài.
    """
    try:
        import h3 as h3lib
        all_h3_set = set(all_h3)
        nbr_r1, nbr_r2 = {}, {}
        for cell in all_h3:
            disk1 = set(h3_grid_disk_compat(h3lib, cell, 1))
            disk2 = set(h3_grid_disk_compat(h3lib, cell, 2))
            nbr_r1[cell] = list((disk1 - {cell}) & all_h3_set)
            nbr_r2[cell] = list((disk2 - disk1)  & all_h3_set)
        avg_r1 = np.mean([len(v) for v in nbr_r1.values()])
        log.info("[ML] H3 neighbors built. Avg ring-1 in dataset: %.1f", avg_r1)
        return nbr_r1, nbr_r2, True
    except ImportError:
        log.warning("[ML] h3 library not found – spatial spillover features skipped. "
                    "Install with: pip install h3")
        return {}, {}, False


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FEATURE ENGINEERING v11
# ─────────────────────────────────────────────────────────────────────────────

def build_features(
    daily: pd.DataFrame,
    all_h3: np.ndarray,
    nbr_r1: dict,
    nbr_r2: dict,
    has_h3: bool,
    train_dates: list,
) -> pd.DataFrame:
    """
    Feature engineering v20 – RAM-optimised rewrite.

    Thay đổi so với bản gốc:
      - float32 thay float64 cho mọi cột số mới tạo ra (giảm ~50% RAM)
      - del + gc.collect() sau mỗi block tốn nhiều bộ nhớ
      - 4G: pivot dùng float32, tính nbr tuần tự rồi xoá ngay, không giữ
            5 ma trận (4973×391) cùng lúc trong RAM
      - 4G: _lag_melt trả về float32 ngay tại chỗ
      - Bỏ biến trung gian không cần thiết
    """
    df = daily.sort_values(["h3_index", "date"]).reset_index(drop=True).copy()
    g  = df.groupby("h3_index", group_keys=False)

    # ── 4A  Own-cell lag / rolling ────────────────────────────────────────
    log.info("[ML]   [4A] Own-cell lag/rolling features")
    for lag in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]:
        df[f"lag_{lag}"] = g["total_crimes"].shift(lag).astype("float32")

    for _col, prefix in [
        ("is_violent",   "violent"),
        ("is_property",  "property"),
        ("is_theft",     "theft"),
        ("is_battery",   "battery"),
        ("is_narcotics", "narcotics"),
    ]:
        for lag in [1, 2, 3, 7, 14]:
            df[f"{prefix}_lag_{lag}"] = g[_col].shift(lag).astype("float32")

    for w in [3, 5, 7, 14, 21, 30, 45, 60, 90]:
        shifted = g["total_crimes"].shift(1)
        roll    = shifted.groupby(df["h3_index"]).rolling(w, min_periods=2)
        df[f"roll_mean_{w}"] = roll.mean().reset_index(level=0, drop=True).astype("float32")
        df[f"roll_sum_{w}"]  = roll.sum().reset_index(level=0, drop=True).astype("float32")
        df[f"roll_std_{w}"]  = roll.std().reset_index(level=0, drop=True).astype("float32")
        df[f"roll_max_{w}"]  = roll.max().reset_index(level=0, drop=True).astype("float32")

    for span in [7, 14, 30, 60, 90]:
        df[f"ewm_{span}"] = (
            g["total_crimes"]
            .shift(1)
            .groupby(df["h3_index"])
            .ewm(span=span, adjust=False, min_periods=2)
            .mean()
            .reset_index(level=0, drop=True)
            .astype("float32")
        )

    for w in [7, 14, 30, 60]:
        df[f"active_rate_{w}"] = (
            g["crime_today"]
            .shift(1)
            .groupby(df["h3_index"])
            .rolling(w, min_periods=2)
            .mean()
            .reset_index(level=0, drop=True)
            .astype("float32")
        )

    # ── 4B  Burst / streak ────────────────────────────────────────────────
    log.info("[ML]   [4B] Burst/streak features")

    def zero_streak(s: pd.Series) -> pd.Series:
        out, cnt = [], 0
        for v in s:
            cnt = cnt + 1 if v == 0 else 0
            out.append(cnt)
        return pd.Series(out, index=s.index)

    df["zero_today"]  = (df["total_crimes"] == 0).astype("int8")
    df["prev_crime"]  = g["crime_today"].shift(1).astype("float32")
    df["zero_streak"] = g["total_crimes"].transform(zero_streak).astype("float32")
    df["burst_3_30"]  = safe_ratio(df["roll_mean_3"],  df["roll_mean_30"],  default=0, clip=10).astype("float32")
    df["burst_7_30"]  = safe_ratio(df["roll_mean_7"],  df["roll_mean_30"],  default=0, clip=10).astype("float32")
    df["burst_14_60"] = safe_ratio(df["roll_mean_14"], df["roll_mean_60"],  default=0, clip=10).astype("float32")
    df["crime_z_30"]  = safe_ratio(
        df["total_crimes"] - df["roll_mean_30"].fillna(0),
        df["roll_std_30"].fillna(0), default=0, clip=10,
    ).astype("float32")
    df["crime_z_90"]  = safe_ratio(
        df["total_crimes"] - df["roll_mean_90"].fillna(0),
        df["roll_std_90"].fillna(0), default=0, clip=10,
    ).astype("float32")

    # v20: short-term shock / emerging activation features.
    df["shock_lag1_vs_roll7"]  = safe_ratio(df["lag_1"], df["roll_mean_7"] + 1e-3, default=0, clip=20).astype("float32")
    df["shock_lag1_vs_roll30"] = safe_ratio(df["lag_1"], df["roll_mean_30"] + 1e-3, default=0, clip=20).astype("float32")
    df["shock_3_14"]           = safe_ratio(df["roll_mean_3"], df["roll_mean_14"] + 1e-3, default=0, clip=10).astype("float32")
    df["shock_3_60"]           = safe_ratio(df["roll_mean_3"], df["roll_mean_60"] + 1e-3, default=0, clip=10).astype("float32")
    df["roll3_minus_roll30"]   = (df["roll_mean_3"].fillna(0) - df["roll_mean_30"].fillna(0)).astype("float32")
    df["roll7_minus_roll60"]   = (df["roll_mean_7"].fillna(0) - df["roll_mean_60"].fillna(0)).astype("float32")
    df["local_spike_z_30"]     = safe_ratio(
        df["lag_1"].fillna(0) - df["roll_mean_30"].fillna(0),
        df["roll_std_30"].fillna(0) + 1e-3, default=0, clip=10,
    ).astype("float32")
    df["local_sudden_activation_3d"] = (
        (df["lag_1"].fillna(0) > 0) & (df["lag_2"].fillna(0) <= 0) & (df["lag_3"].fillna(0) <= 0)
    ).astype("int8")
    df["local_recent_decay"] = (
        0.55 * df["lag_1"].fillna(0).astype("float32") +
        0.30 * df["lag_2"].fillna(0).astype("float32") +
        0.15 * df["lag_3"].fillna(0).astype("float32")
    ).astype("float32")
    df["active_recent_vs_long"] = safe_ratio(
        df["active_rate_7"], df["active_rate_60"] + 1e-3, default=0, clip=10
    ).astype("float32")

    # ── 4C  Crime-type rolling composition ───────────────────────────────
    log.info("[ML]   [4C] Crime-type composition")
    for col_name, prefix in [
        ("is_violent",   "violent"),
        ("is_property",  "property"),
        ("is_theft",     "theft"),
        ("is_battery",   "battery"),
        ("is_narcotics", "narcotics"),
    ]:
        shifted = g[col_name].shift(1)
        for w in [7, 14, 30]:
            rs = shifted.groupby(df["h3_index"]).rolling(w, min_periods=2).sum()
            df[f"{prefix}_roll_sum_{w}"]   = rs.reset_index(level=0, drop=True).astype("float32")
            df[f"{prefix}_roll_ratio_{w}"] = safe_ratio(
                df[f"{prefix}_roll_sum_{w}"], df[f"roll_sum_{w}"], default=0, clip=1
            ).astype("float32")

    # ── 4D  Weather / context ────────────────────────────────────────────
    log.info("[ML]   [4D] Weather/context features")
    for c in ["temp_max", "temp_min", "precipitation", "wind_speed"]:
        if c in df.columns:
            df[f"{c}_lag1"]  = g[c].shift(1).astype("float32")
            df[f"{c}_lag7"]  = g[c].shift(7).astype("float32")
            df[f"{c}_roll7"] = (
                g[c].shift(1)
                .groupby(df["h3_index"])
                .rolling(7, min_periods=2)
                .mean()
                .reset_index(level=0, drop=True)
                .astype("float32")
            )
            df[f"{c}_diff1"] = (df[c] - df[f"{c}_lag1"]).astype("float32")

    # ── 4E  Community-level temporal ─────────────────────────────────────
    log.info("[ML]   [4E] Community-level features")
    if "community_area" in df.columns:
        df["community_area"] = (
            pd.to_numeric(df["community_area"], errors="coerce").fillna(-1).astype(int)
        )
    else:
        df["community_area"] = -1

    comm_daily = (
        df.groupby(["community_area", "date"], as_index=False)
        .agg(
            comm_total    = ("total_crimes", "sum"),
            comm_active   = ("crime_today",  "sum"),
            comm_violent  = ("is_violent",   "sum"),
            comm_property = ("is_property",  "sum"),
            comm_arrests  = ("arrest_num",   "sum"),
        )
    )
    cg = comm_daily.groupby("community_area", group_keys=False)
    for w in [7, 14, 30, 60]:
        shifted = cg["comm_total"].shift(1)
        comm_daily[f"comm_roll_mean_{w}"] = (
            shifted.groupby(comm_daily["community_area"])
            .rolling(w, min_periods=2).mean()
            .reset_index(level=0, drop=True)
            .astype("float32")
        )
        comm_daily[f"comm_roll_sum_{w}"] = (
            shifted.groupby(comm_daily["community_area"])
            .rolling(w, min_periods=2).sum()
            .reset_index(level=0, drop=True)
            .astype("float32")
        )
    comm_daily["comm_burst_7_30"] = safe_ratio(
        comm_daily["comm_roll_mean_7"], comm_daily["comm_roll_mean_30"], default=0, clip=10
    ).astype("float32")
    n_h3_per_comm = df.groupby("community_area")["h3_index"].nunique().reset_index()
    n_h3_per_comm.columns = ["community_area", "n_h3_in_comm"]
    comm_daily = comm_daily.merge(n_h3_per_comm, on="community_area", how="left")
    comm_daily["comm_active_rate"] = safe_ratio(
        comm_daily["comm_active"], comm_daily["n_h3_in_comm"], default=0, clip=1
    ).astype("float32")
    drop_cols = ["comm_total", "comm_active", "comm_violent", "comm_property", "comm_arrests"]
    df = df.merge(comm_daily.drop(columns=drop_cols), on=["community_area", "date"], how="left")
    del comm_daily, n_h3_per_comm
    gc.collect()

    # ── 4F  City-level context ───────────────────────────────────────────
    log.info("[ML]   [4F] City-level features")
    city_daily = (
        df.groupby("date", as_index=False)
        .agg(
            city_total    = ("total_crimes", "sum"),
            city_active   = ("crime_today",  "sum"),
            city_violent  = ("is_violent",   "sum"),
            city_property = ("is_property",  "sum"),
            city_arrests  = ("arrest_num",   "sum"),
        )
    )
    for c in ["city_total", "city_active", "city_violent", "city_property"]:
        city_daily[f"{c}_lag1"] = city_daily[c].shift(1).astype("float32")
        for w in [7, 14, 30]:
            city_daily[f"{c}_roll_mean_{w}"] = (
                city_daily[c].shift(1).rolling(w, min_periods=2).mean().astype("float32")
            )
    city_daily["city_burst_7_30"] = safe_ratio(
        city_daily["city_total_roll_mean_7"],
        city_daily["city_total_roll_mean_30"],
        default=0, clip=10,
    ).astype("float32")
    drop_city = ["city_total", "city_active", "city_violent", "city_property", "city_arrests"]
    df = df.merge(city_daily.drop(columns=drop_city), on="date", how="left")
    df["h3_city_share_7"]  = safe_ratio(
        df["roll_mean_7"],  df["city_total_roll_mean_7"]  + 1e-3, default=0, clip=1
    ).astype("float32")
    df["h3_city_share_30"] = safe_ratio(
        df["roll_mean_30"], df["city_total_roll_mean_30"] + 1e-3, default=0, clip=1
    ).astype("float32")
    del city_daily
    gc.collect()

    # ── 4G  Spatial Spillover (H3 neighbors) ─────────────────────────────
    log.info("[ML]   [4G] H3 spatial spillover v20 target-safe")
    if has_h3:
        dates_order = pd.Index(sorted(daily["date"].unique()))
        cells_order = pd.Index(sorted(daily["h3_index"].unique()))
        n_dates, n_cells = len(dates_order), len(cells_order)
        cell_to_pos = {c: i for i, c in enumerate(cells_order)}

        crime_pivot = (
            daily.pivot(index="date", columns="h3_index", values="total_crimes")
            .reindex(index=dates_order, columns=cells_order)
            .fillna(0).astype("float32")
        )
        active_pivot = (
            daily.pivot(index="date", columns="h3_index", values="crime_today")
            .reindex(index=dates_order, columns=cells_order)
            .fillna(0).astype("float32")
        )
        crime_arr = np.ascontiguousarray(crime_pivot.to_numpy(dtype=np.float32))
        active_arr = np.ascontiguousarray(active_pivot.to_numpy(dtype=np.float32))
        del crime_pivot, active_pivot
        gc.collect()

        r1_idx = [[cell_to_pos[x] for x in nbr_r1.get(cell, []) if x in cell_to_pos] for cell in cells_order]
        r2_idx = [[cell_to_pos[x] for x in nbr_r2.get(cell, []) if x in cell_to_pos] for cell in cells_order]

        # Column-major reshape below matches h3 repeated outer / dates tiled inner.
        r1_neighbor_counts = np.array([len(x) for x in r1_idx], dtype=np.float32)
        r2_neighbor_counts = np.array([len(x) for x in r2_idx], dtype=np.float32)
        nbr_long = pd.DataFrame({
            "date": np.tile(dates_order.values, n_cells),
            "h3_index": np.repeat(cells_order.values.astype(str), n_dates),
        })
        nbr_long["nbr_r1_neighbor_count"] = np.repeat(r1_neighbor_counts, n_dates).astype(np.float32)
        nbr_long["nbr_r2_neighbor_count"] = np.repeat(r2_neighbor_counts, n_dates).astype(np.float32)

        def lag1_matrix(mat: np.ndarray) -> np.ndarray:
            out = np.zeros_like(mat, dtype=np.float32)
            out[1:, :] = mat[:-1, :]
            return out

        def add_neighbor_feature(name: str, source_arr: np.ndarray, ring_idx: list[list[int]], stat: str):
            mat = np.zeros((n_dates, n_cells), dtype=np.float32)
            for j, idxs in enumerate(ring_idx):
                if not idxs:
                    continue
                sub = source_arr[:, idxs]
                if stat == "mean":
                    mat[:, j] = sub.mean(axis=1, dtype=np.float32)
                elif stat == "sum":
                    mat[:, j] = sub.sum(axis=1, dtype=np.float32)
                elif stat == "max":
                    mat[:, j] = sub.max(axis=1)
                elif stat == "std":
                    mat[:, j] = sub.std(axis=1, dtype=np.float32)
                else:
                    raise ValueError(stat)

            if name == "nbr_r1_roll7":
                shifted = lag1_matrix(mat)
                feat = (
                    pd.DataFrame(shifted, index=dates_order, columns=cells_order)
                    .rolling(7, min_periods=2)
                    .mean().fillna(0).to_numpy(dtype=np.float32)
                )
                del shifted
            else:
                feat = lag1_matrix(mat)

            nbr_long[name] = feat.reshape(-1, order="F").astype(np.float32)
            del mat, feat
            gc.collect()

        log.info("[ML]     computing nbr_r1 mean/sum/max/std/active-count + r2 + roll7 ...")
        add_neighbor_feature("nbr_r1_mean", crime_arr, r1_idx, "mean")
        add_neighbor_feature("nbr_r1_sum",  crime_arr, r1_idx, "sum")
        add_neighbor_feature("nbr_r1_max",  crime_arr, r1_idx, "max")
        add_neighbor_feature("nbr_r1_std",  crime_arr, r1_idx, "std")
        add_neighbor_feature("nbr_r1_active_rate", active_arr, r1_idx, "mean")
        add_neighbor_feature("nbr_r1_active_count", active_arr, r1_idx, "sum")
        add_neighbor_feature("nbr_r1_any_active", active_arr, r1_idx, "max")
        add_neighbor_feature("nbr_r2_mean", crime_arr, r2_idx, "mean")
        add_neighbor_feature("nbr_r1_roll7", crime_arr, r1_idx, "mean")

        df = df.merge(nbr_long, on=["date", "h3_index"], how="left")
        base_nbr_cols = [
            "nbr_r1_mean", "nbr_r1_sum", "nbr_r1_max", "nbr_r1_std",
            "nbr_r1_active_rate", "nbr_r1_active_count", "nbr_r1_any_active",
            "nbr_r2_mean", "nbr_r1_roll7", "nbr_r1_neighbor_count", "nbr_r2_neighbor_count",
        ]
        for c in base_nbr_cols:
            df[c] = df[c].fillna(0).astype("float32")

        df["h3_vs_nbr_r1"] = safe_ratio(
            df["roll_mean_7"], df["nbr_r1_mean"] + 1e-3, default=0, clip=20
        ).astype("float32")
        df["h3_above_nbr"] = (df["lag_1"] > df["nbr_r1_mean"]).astype("int8")
        df["nbr_contagion"] = (df["nbr_r1_active_rate"] > 0.5).astype("int8")
        df["nbr_r1_burst_1_7"] = safe_ratio(
            df["nbr_r1_mean"], df["nbr_r1_roll7"] + 1e-3, default=0, clip=10
        ).astype("float32")
        df["h3_nbr_joint_activity"] = (
            df["active_rate_7"].fillna(0).astype("float32") * df["nbr_r1_active_rate"]
        ).astype("float32")
        df["nbr_r1_vs_r2"] = safe_ratio(
            df["nbr_r1_mean"], df["nbr_r2_mean"] + 1e-3, default=0, clip=10
        ).astype("float32")

        sg = df.groupby("h3_index", group_keys=False)
        df["nbr_r1_active_roll7"] = (
            sg["nbr_r1_active_rate"].transform(lambda s: s.rolling(7, min_periods=2).mean())
            .fillna(0).astype("float32")
        )
        df["nbr_r1_active_burst"] = safe_ratio(
            df["nbr_r1_active_rate"], df["nbr_r1_active_roll7"] + 1e-3, default=0, clip=10
        ).astype("float32")
        df["nbr_r1_new_activation"] = (
            (df["nbr_r1_active_rate"] > 0) &
            (sg["nbr_r1_active_rate"].shift(1).fillna(0) <= 0)
        ).astype("int8")
        df["nbr_r1_hot_count_x_local_cold"] = (
            df["nbr_r1_active_count"].fillna(0).astype("float32") *
            (df["lag_1"].fillna(0) <= 0).astype("float32")
        ).astype("float32")
        df["nbr_max_vs_local_lag1"] = safe_ratio(
            df["nbr_r1_max"], df["lag_1"].fillna(0) + 1e-3, default=0, clip=20
        ).astype("float32")
        df["nbr_dispersion_signal"] = safe_ratio(
            df["nbr_r1_std"], df["nbr_r1_mean"] + 1e-3, default=0, clip=10
        ).astype("float32")
        df["local_nbr_double_burst"] = (
            df["burst_3_30"].fillna(0).clip(0, 10).astype("float32") *
            df["nbr_r1_burst_1_7"].fillna(0).clip(0, 10).astype("float32")
        ).astype("float32")

        df["nbr_r1_spike_vs_roll7"] = safe_ratio(
            df["nbr_r1_mean"].fillna(0) - df["nbr_r1_roll7"].fillna(0),
            df["nbr_r1_std"].fillna(0) + 1e-3, default=0, clip=10,
        ).astype("float32")
        df["nbr_r1_active_accel"] = (
            df["nbr_r1_active_rate"].fillna(0).astype("float32") -
            sg["nbr_r1_active_rate"].shift(1).fillna(0).astype("float32")
        ).astype("float32")
        df["nbr_any_active_x_local_cold"] = (
            df["nbr_r1_any_active"].fillna(0).astype("float32") *
            (df["lag_1"].fillna(0) <= 0).astype("float32")
        ).astype("float32")
        df["nbr_spike_x_local_sudden"] = (
            (df["nbr_r1_spike_vs_roll7"].fillna(0) > 1.0).astype("float32") *
            df.get("local_sudden_activation_3d", 0)
        ).astype("float32")
        df["nbr_hot_count_per_neighbor"] = safe_ratio(
            df["nbr_r1_active_count"],
            df["nbr_r1_neighbor_count"].replace(0, np.nan),
            default=0,
            clip=1,
        ).astype("float32")

        del nbr_long, crime_arr, active_arr, r1_idx, r2_idx, r1_neighbor_counts, r2_neighbor_counts
        gc.collect()
        log.info("[ML]     ✓ Spatial spillover features added, single-merge RAM-safe")
    else:
        for c in [
            "nbr_r1_mean", "nbr_r1_sum", "nbr_r1_max", "nbr_r1_std",
            "nbr_r1_active_rate", "nbr_r1_active_count", "nbr_r1_any_active",
            "nbr_r2_mean", "nbr_r1_roll7", "h3_vs_nbr_r1", "h3_above_nbr",
            "nbr_contagion", "nbr_r1_burst_1_7", "h3_nbr_joint_activity",
            "nbr_r1_vs_r2", "nbr_r1_active_roll7", "nbr_r1_active_burst",
            "nbr_r1_new_activation", "nbr_r1_hot_count_x_local_cold",
            "nbr_max_vs_local_lag1", "nbr_dispersion_signal", "local_nbr_double_burst",
        ]:
            df[c] = np.float32(0)
        log.warning("[ML]     ⚠️ Spatial spillover skipped (h3 not installed)")

    # ── 4H  Train-only spatial priors ─────────────────────────────────────
    log.info("[ML]   [4H] Train-only spatial priors")
    train_mask_prior = df["date"].isin(train_dates)
    h3_prior = (
        df.loc[train_mask_prior]
        .groupby("h3_index")["total_crimes"]
        .agg(["mean", "std", "sum", "max"])
        .reset_index()
    )
    h3_prior.columns = [
        "h3_index", "h3_train_mean", "h3_train_std",
        "h3_train_sum", "h3_train_max",
    ]
    h3_prior["h3_train_percentile"]  = h3_prior["h3_train_mean"].rank(pct=True).astype("float32")
    h3_active = (
        df.loc[train_mask_prior]
        .groupby("h3_index", as_index=False)["crime_today"]
        .mean()
        .rename(columns={"crime_today": "h3_train_active_rate"})
    )
    h3_prior = h3_prior.merge(h3_active, on="h3_index", how="left")
    df = df.merge(h3_prior, on="h3_index", how="left")
    del h3_prior

    comm_prior = (
        df.loc[train_mask_prior]
        .groupby("community_area")["total_crimes"]
        .agg(["mean", "std", "sum", "max"])
        .reset_index()
    )
    comm_prior.columns = [
        "community_area", "comm_train_mean", "comm_train_std",
        "comm_train_sum", "comm_train_max",
    ]
    comm_prior["comm_train_percentile"] = comm_prior["comm_train_mean"].rank(pct=True).astype("float32")
    df = df.merge(comm_prior, on="community_area", how="left")
    del comm_prior
    gc.collect()

    for c in [
        "h3_train_mean", "h3_train_std", "h3_train_sum", "h3_train_max",
        "h3_train_percentile", "h3_train_active_rate",
        "comm_train_mean", "comm_train_std", "comm_train_sum",
        "comm_train_max", "comm_train_percentile",
    ]:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype("float32")

    df["h3_train_pct"] = df["h3_train_percentile"].astype("float32")

    # ── 4I  Near-Repeat Victimization ─────────────────────────────────────
    log.info("[ML]   [4I] Near-repeat victimization (Near-Repeat Theory)")
    df["had_crime_1d"]   = (g["total_crimes"].shift(1) > 0).astype("int8")
    df["had_crime_2d"]   = (g["total_crimes"].shift(2) > 0).astype("int8")
    df["had_crime_3d"]   = (g["total_crimes"].shift(3) > 0).astype("int8")
    df["near_repeat_3d"] = (df["had_crime_1d"] + df["had_crime_2d"] + df["had_crime_3d"]).astype("int8")
    df["near_repeat_5d"] = sum(
        [(g["total_crimes"].shift(i) > 0).astype(int) for i in range(1, 6)]
    ).astype("int8")
    df["reactivation"] = (
        (df["zero_streak"] >= 3) & (g["total_crimes"].shift(1) > 0)
    ).astype("int8")

    # ── 4J  Cross-Crime Lead-Lag ───────────────────────────────────────────
    log.info("[ML]   [4J] Cross-crime lead-lag (Escalation Theory)")
    df["property_lead_violent"]    = (
        g["is_property"].shift(1) * (g["is_violent"].shift(1) == 0)
    ).astype("float32")
    df["narcotics_before_violent"] = (g["is_narcotics"].shift(1) > 0).astype("int8")
    if "violent_lag_1" in df.columns and "property_lag_1" in df.columns:
        df["violent_x_property_lag1"] = (
            df["violent_lag_1"] * df["property_lag_1"]
        ).astype("float32")
    else:
        df["violent_x_property_lag1"] = np.float32(0)

    # ── 4K  Arrest Deterrence ─────────────────────────────────────────────
    log.info("[ML]   [4K] Arrest deterrence (Deterrence Theory)")
    df["arrest_roll_7"]  = (
        g["arrest_num"].shift(1)
        .groupby(df["h3_index"])
        .rolling(7,  min_periods=2).sum()
        .reset_index(level=0, drop=True)
        .astype("float32")
    )
    df["arrest_roll_30"] = (
        g["arrest_num"].shift(1)
        .groupby(df["h3_index"])
        .rolling(30, min_periods=2).sum()
        .reset_index(level=0, drop=True)
        .astype("float32")
    )
    df["arrest_rate_7"]  = safe_ratio(
        df["arrest_roll_7"],  df["roll_sum_7"]  + 1e-3, default=0, clip=1
    ).astype("float32")
    df["arrest_rate_30"] = safe_ratio(
        df["arrest_roll_30"], df["roll_sum_30"] + 1e-3, default=0, clip=1
    ).astype("float32")
    df["deterrence_signal"] = (
        df["arrest_rate_7"] * (1 - df["burst_7_30"].clip(0, 1))
    ).astype("float32")

    # ── 4L  Density-Normalized Crime ──────────────────────────────────────
    log.info("[ML]   [4L] Density-normalized crime")
    if "population_density" in df.columns:
        pop = df["population_density"].clip(lower=1)
        df["crime_per_1000pop"]       = safe_ratio(
            df["total_crimes"] * 1000, pop, default=0, clip=50
        ).astype("float32")
        df["crime_roll7_per_1000pop"] = safe_ratio(
            df["roll_mean_7"]  * 1000, pop, default=0, clip=50
        ).astype("float32")
        city_avg_rate = df["total_crimes"].sum() / max(pop.sum(), 1)
        df["crime_vs_density_expected"] = (
            df["total_crimes"] - (pop * city_avg_rate)
        ).astype("float32")
    else:
        df["crime_per_1000pop"] = df["total_crimes"].astype("float32")

    # ── 4M  Holiday / Seasonal Events ─────────────────────────────────────
    log.info("[ML]   [4M] Holiday/event features (Routine Activity Theory)")
    hol_df = holiday_features(df["date"], ALL_HOLIDAYS)
    # holiday_features trả về float64 — downcast luôn
    for c in hol_df.columns:
        hol_df[c] = hol_df[c].astype("float32")
    df = pd.concat([df, hol_df], axis=1)
    del hol_df
    gc.collect()

    # ── 4N  Momentum + Trend Direction ────────────────────────────────────
    log.info("[ML]   [4N] Momentum and trend direction")
    df["trend_slope_7_14"]  = safe_ratio(
        df["roll_mean_7"]  - df["roll_mean_14"],
        df["roll_mean_14"] + 1e-3, default=0, clip=5,
    ).astype("float32")
    df["trend_slope_14_30"] = safe_ratio(
        df["roll_mean_14"] - df["roll_mean_30"],
        df["roll_mean_30"] + 1e-3, default=0, clip=5,
    ).astype("float32")
    df["trend_direction"]    = np.sign(df["trend_slope_7_14"]).astype("float32")
    df["crime_acceleration"] = (df["trend_slope_7_14"] - df["trend_slope_14_30"]).astype("float32")
    df["same_dow_roll8"] = (
        df.assign(_c=df["total_crimes"], _dow=df["dayofweek"])
        .groupby(["h3_index", "_dow"])["_c"]
        .transform(lambda s: s.shift(1).rolling(8, min_periods=2).mean())
        .astype("float32")
    )
    df["same_dow_vs_overall"] = safe_ratio(
        df["same_dow_roll8"], df["roll_mean_30"] + 1e-3, default=0, clip=5
    ).astype("float32")

    # ── 4O  Hardship Interaction ───────────────────────────────────────────
    log.info("[ML]   [4O] Hardship interaction (Social Disorganization Theory)")
    if "hardship_index" in df.columns:
        hi = df["hardship_index"].fillna(0).astype("float32")
        df["hardship_x_roll7"]       = (hi * df["roll_mean_7"].fillna(0)).astype("float32")
        df["hardship_x_active_rate"] = (
            hi * df.get("active_rate_7", pd.Series(0, index=df.index)).fillna(0)
        ).astype("float32")
        df["hardship_x_nbr_r1"]      = (hi * df["nbr_r1_mean"].fillna(0)).astype("float32")
    if "unemployment_rate" in df.columns:
        unemp = df["unemployment_rate"].fillna(0).astype("float32")
        df["unemployment_x_reactivation"] = (unemp * df["reactivation"].fillna(0)).astype("float32")
        df["unemployment_x_zero_streak"]  = (
            unemp * (df["zero_streak"] > 7).astype("float32")
        ).astype("float32")

    # ── 4P  Prior-relative features ───────────────────────────────────────
    log.info("[ML]   [4P] Prior-relative features")
    df["h3_vs_comm_prior"]     = safe_ratio(
        df["h3_train_mean"], df["comm_train_mean"] + 1e-3, default=0, clip=20
    ).astype("float32")
    df["h3_recent_vs_comm"]    = safe_ratio(
        df["roll_mean_7"],
        df.get("comm_roll_mean_7", pd.Series(1, index=df.index)) + 1e-3,
        default=0, clip=20,
    ).astype("float32")
    df["high_prior_x_weekend"] = (df["h3_train_percentile"] * df["is_weekend"]).astype("float32")
    df["high_prior_x_friday"]  = (df["h3_train_percentile"] * df["is_friday"]).astype("float32")
    df["high_prior_x_holiday"] = (
        df["h3_train_percentile"] * df.get("is_holiday", 0)
    ).astype("float32")
    df["active_rate_vs_train"] = safe_ratio(
        df.get("active_rate_7", df["roll_mean_7"]),
        df["h3_train_active_rate"] + 1e-3,
        default=0, clip=10,
    ).astype("float32")

    log.info("[ML] Feature engineering done. Shape: %s", df.shape)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TRAIN + EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    df: pd.DataFrame,
    train_dates: list,
    calib_dates: list,
    val_dates:   list,
    test_dates:  list,
) -> tuple:
    """
    Train v24 fixed GroupBlend ranker.

    The final score is a weighted blend of three LightGBM group models:
      score = 0.1 * temporal + 0.6 * spatial + 0.3 * context

    Weights are fixed from Kaggle Run 3 GroupBlend_VAL_best and are not selected
    on TEST. Threshold remains diagnostic; dashboard uses top-k rank bands.
    Returns: (artifact, feature_cols, train_medians, threshold, val_res, test_res, sample_weight_info)
    """
    try:
        import lightgbm as lgb
    except Exception as e:
        raise RuntimeError("LightGBM is required for v24 GroupBlend. Install lightgbm in the Airflow image.") from e

    log.info("[ML] Preparing train/calib/val/test splits for v24 GroupBlend ranker...")

    ID_COLS = {"h3_index", "date"}
    TARGET_COLS = {"next_total", "next_has_crime", "next_hotspot", "next_topk_group", "next_rank_pct"}
    exclude = ID_COLS | TARGET_COLS

    feature_cols = [
        c for c in df.columns
        if c not in exclude and not c.startswith("next_") and pd.api.types.is_numeric_dtype(df[c])
    ]

    supervised = df["next_total"].notna()
    train_mask = df["date"].isin(train_dates) & supervised
    calib_mask = df["date"].isin(calib_dates) & supervised
    val_mask   = df["date"].isin(val_dates)   & supervised
    test_mask  = df["date"].isin(test_dates)  & supervised

    redundant_priors = {"h3_train_sum", "h3_train_max", "h3_train_pct", "comm_train_sum", "comm_train_max"}
    strong_raw_priors = {"h3_train_mean", "h3_train_std", "comm_train_mean", "comm_train_std"}
    long_memory_sums = {"roll_sum_60", "roll_sum_90"}
    direct_static = {"community_area", "latitude", "longitude", "day_of_week", "month", "median_income", "per_capita_income"}
    removed = set()
    if DROP_REDUNDANT_PRIOR_FEATURES:
        removed |= redundant_priors
    if DROP_STRONG_RAW_PRIORS:
        removed |= strong_raw_priors
    if DROP_REDUNDANT_LONG_MEMORY_SUMS:
        removed |= long_memory_sums
    if DROP_DIRECT_STATIC_CONTEXT_FEATURES:
        removed |= direct_static
    if DROP_AUDIT_ONLY_PRIOR_BAND_FEATURES:
        removed |= AUDIT_ONLY_PRIOR_BAND_FEATURES
    feature_cols = [c for c in feature_cols if c not in removed]
    log.info("[ML] Removed guarded/redundant features: %d", len(removed))

    keep, dropped = [], []
    for c in feature_cols:
        s = pd.to_numeric(df.loc[train_mask, c], errors="coerce")
        if s.notna().sum() > 0 and s.nunique(dropna=True) > 1:
            keep.append(c)
        else:
            dropped.append(c)
    feature_cols = keep

    X_train, train_medians = sanitize_numeric(df.loc[train_mask], feature_cols)
    X_calib, _ = sanitize_numeric(df.loc[calib_mask], feature_cols, train_medians)
    X_val, _   = sanitize_numeric(df.loc[val_mask], feature_cols, train_medians)
    X_test, _  = sanitize_numeric(df.loc[test_mask], feature_cols, train_medians)

    y_train = df.loc[train_mask, TARGET_COL].astype(int)
    y_calib = df.loc[calib_mask, TARGET_COL].astype(int)
    y_val   = df.loc[val_mask, TARGET_COL].astype(int)
    y_test  = df.loc[test_mask, TARGET_COL].astype(int)

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    raw_scale_pos_weight = neg / max(pos, 1)
    scale_pos_weight = min(CLASS_WEIGHT_CAP, raw_scale_pos_weight ** CLASS_WEIGHT_POWER)
    scale_pos_weight *= CLASS_WEIGHT_EXTRA_BOOST.get(1, 1.0)

    sample_weight = np.ones(len(y_train), dtype=np.float32)
    train_part = df.loc[train_mask]
    pos_mask = y_train.values == 1
    if "local_sudden_activation_3d" in train_part.columns:
        sample_weight[pos_mask & (train_part["local_sudden_activation_3d"].fillna(0).values > 0)] *= EMERGING_POSITIVE_WEIGHT_BOOST
    if "h3_train_percentile" in train_part.columns:
        low_mid = train_part["h3_train_percentile"].fillna(0).values < 0.75
        sample_weight[pos_mask & low_mid] *= LOW_MID_PRIOR_POSITIVE_WEIGHT_BOOST
    if "nbr_r1_spike_vs_roll7" in train_part.columns:
        nbr_spike = train_part["nbr_r1_spike_vs_roll7"].fillna(0).values > 1.0
        sample_weight[pos_mask & nbr_spike] *= NEIGHBOR_SPIKE_POSITIVE_WEIGHT_BOOST
    sample_weight = np.clip(sample_weight, 1.0, MAX_SAMPLE_WEIGHT_MULTIPLIER).astype(np.float32)

    feature_buckets = {
        "temporal": [c for c in feature_cols if feature_blend_bucket(c) == "temporal"],
        "spatial":  [c for c in feature_cols if feature_blend_bucket(c) == "spatial"],
        "context":  [c for c in feature_cols if feature_blend_bucket(c) == "context"],
    }
    for gname in ["temporal", "spatial", "context"]:
        if len(feature_buckets[gname]) < 5:
            raise ValueError(f"Not enough features for {gname} group: {len(feature_buckets[gname])}")

    log.info("[ML] Train=%d Calib=%d Val=%d Test=%d features=%d dropped=%d",
             len(X_train), len(X_calib), len(X_val), len(X_test), len(feature_cols), len(dropped))
    log.info("[ML] Group features: temporal=%d spatial=%d context=%d",
             len(feature_buckets["temporal"]), len(feature_buckets["spatial"]), len(feature_buckets["context"]))
    log.info("[ML] Hotspot train rate=%.3f%% | scale_pos_weight raw=%.2f used=%.2f",
             y_train.mean()*100, raw_scale_pos_weight, scale_pos_weight)

    lgbm_params = dict(
        n_estimators=2200, learning_rate=0.025, num_leaves=23, max_depth=-1,
        min_child_samples=30, subsample=0.90, colsample_bytree=0.80,
        reg_alpha=0.20, reg_lambda=4.0, objective="binary",
        random_state=RANDOM_STATE, n_jobs=-1, class_weight=None, verbosity=-1,
    )

    group_models = {}
    group_scores = {"calib": {}, "val": {}, "test": {}}
    group_thresholds = {}

    for group_name, cols in feature_buckets.items():
        log.info("[ML] Training LGBM_%s with %d features", group_name, len(cols))
        model_g = lgb.LGBMClassifier(**lgbm_params)
        try:
            model_g.fit(
                X_train[cols], y_train,
                sample_weight=sample_weight,
                eval_set=[(X_calib[cols], y_calib)],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
            )
        except TypeError:
            model_g.fit(X_train[cols], y_train, sample_weight=sample_weight, eval_set=[(X_calib[cols], y_calib)], eval_metric="average_precision")

        calib_s = model_g.predict_proba(X_calib[cols])[:, 1].astype(np.float32)
        val_s   = model_g.predict_proba(X_val[cols])[:, 1].astype(np.float32)
        test_s  = model_g.predict_proba(X_test[cols])[:, 1].astype(np.float32)
        thr_g, calib_f1_g, _ = find_best_threshold_guarded(y_calib.values, calib_s)
        group_models[group_name] = {"model": model_g, "features": cols, "threshold": float(thr_g)}
        group_scores["calib"][group_name] = calib_s
        group_scores["val"][group_name] = val_s
        group_scores["test"][group_name] = test_s
        group_thresholds[group_name] = float(thr_g)
        log.info("[ML] LGBM_%s threshold=%.3f calib_f1=%.4f", group_name, thr_g, calib_f1_g)

    def blend(split_name: str) -> np.ndarray:
        return (
            GROUP_BLEND_WEIGHTS["temporal"] * group_scores[split_name]["temporal"]
            + GROUP_BLEND_WEIGHTS["spatial"] * group_scores[split_name]["spatial"]
            + GROUP_BLEND_WEIGHTS["context"] * group_scores[split_name]["context"]
        ).astype(np.float32)

    calib_score = blend("calib")
    val_score = blend("val")
    test_score = blend("test")

    if USE_PRED_RATE_GUARD:
        best_t, best_f1_calib, _ = find_best_threshold_guarded(y_calib.values, calib_score)
    else:
        best_t, best_f1_calib = find_best_threshold(y_calib.values, calib_score)
    log.info("[ML] GroupBlend diagnostic threshold=%.3f | F1_CALIB=%.4f | weights=%s", best_t, best_f1_calib, GROUP_BLEND_WEIGHTS)

    val_res = evaluate_ranker(df.loc[val_mask], y_val.values, val_score, best_t)
    test_res = evaluate_ranker(df.loc[test_mask], y_test.values, test_score, best_t)

    log.info("[ML] VAL  – AUC=%.4f AP=%.4f F1=%.4f Top05_R=%.4f Top05_L=%.2fx Top15_R=%.4f Top30_R=%.4f",
             val_res["auc"], val_res["ap"], val_res["f1"], val_res["top05_recall"], val_res["top05_lift"], val_res["top15_recall"], val_res["top30_recall"])
    log.info("[ML] TEST – AUC=%.4f AP=%.4f F1=%.4f Top05_R=%.4f Top05_L=%.2fx Top15_R=%.4f Top30_R=%.4f",
             test_res["auc"], test_res["ap"], test_res["f1"], test_res["top05_recall"], test_res["top05_lift"], test_res["top15_recall"], test_res["top30_recall"])

    artifact = {
        "group_models": group_models,
        "group_weights": GROUP_BLEND_WEIGHTS.copy(),
        "feature_buckets": feature_buckets,
        "feature_cols": feature_cols,
        "group_thresholds": group_thresholds,
        "dropped_features": dropped,
    }
    sample_weight_info = {"raw_scale_pos_weight": raw_scale_pos_weight, "used_scale_pos_weight": scale_pos_weight}
    return artifact, feature_cols, train_medians, best_t, val_res, test_res, sample_weight_info


# ─────────────────────────────────────────────────────────────────────────────
# 7.  INFERENCE: NEXT DAY PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _predict_groupblend(df_part: pd.DataFrame, artifact: dict, train_medians: pd.Series) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Predict GroupBlend score and component scores for any dataframe slice."""
    feature_cols = artifact["feature_cols"]
    X, _ = sanitize_numeric(df_part, feature_cols, train_medians)
    group_scores: dict[str, np.ndarray] = {}
    for group_name, info in artifact["group_models"].items():
        cols = info["features"]
        group_scores[group_name] = info["model"].predict_proba(X[cols])[:, 1].astype(np.float32)
    weights = artifact["group_weights"]
    score = (
        weights["temporal"] * group_scores["temporal"]
        + weights["spatial"] * group_scores["spatial"]
        + weights["context"] * group_scores["context"]
    ).astype(np.float32)
    return score, group_scores


def add_confidence_columns(out: pd.DataFrame, group_scores: dict[str, np.ndarray]) -> pd.DataFrame:
    """Add deploy-oriented confidence diagnostics.

    confidence_score combines:
      1. model_agreement: lower std among temporal/spatial/context scores is safer;
      2. rank_margin_confidence: farther from CRITICAL/HIGH/MEDIUM cutoffs is safer;
      3. rank_extremity: very high/very low ranks are easier to defend than boundary cases.
    """
    out = out.copy()
    out["temporal_score"] = group_scores["temporal"].astype(float)
    out["spatial_score"] = group_scores["spatial"].astype(float)
    out["context_score"] = group_scores["context"].astype(float)
    out["group_w_temporal"] = GROUP_BLEND_WEIGHTS["temporal"]
    out["group_w_spatial"] = GROUP_BLEND_WEIGHTS["spatial"]
    out["group_w_context"] = GROUP_BLEND_WEIGHTS["context"]

    stack = np.vstack([group_scores["temporal"], group_scores["spatial"], group_scores["context"]]).T
    out["group_score_std"] = stack.std(axis=1).astype(float)
    out["group_score_range"] = (stack.max(axis=1) - stack.min(axis=1)).astype(float)
    out["model_agreement"] = np.clip(1.0 - 2.0 * out["group_score_std"].values, 0.0, 1.0)

    cutoffs = np.array([DASHBOARD_TOP_CRITICAL, DASHBOARD_TOP_HIGH, DASHBOARD_TOP_MEDIUM], dtype=np.float32)
    rank_pct = out["risk_rank_pct"].to_numpy(dtype=np.float32)
    nearest = np.min(np.abs(rank_pct[:, None] - cutoffs[None, :]), axis=1)
    out["score_margin_to_cutoff"] = nearest.astype(float)
    out["rank_margin_confidence"] = np.clip(nearest / DASHBOARD_TOP_CRITICAL, 0.0, 1.0)
    out["rank_extremity"] = np.clip(np.abs(0.5 - rank_pct) * 2.0, 0.0, 1.0)

    out["confidence_score"] = (
        0.45 * out["model_agreement"].values
        + 0.35 * out["rank_margin_confidence"].values
        + 0.20 * out["rank_extremity"].values
    ).clip(0.0, 1.0)
    out["confidence_level"] = pd.cut(
        out["confidence_score"],
        bins=[-0.01, 0.50, 0.75, 1.01],
        labels=["LOW", "MEDIUM", "HIGH"],
    ).astype(str)
    return out


def build_predictions(
    df: pd.DataFrame,
    artifact: dict,
    train_medians: pd.Series,
    best_threshold: float,
    val_res: dict,
    test_res: dict,
) -> pd.DataFrame:
    """Use latest feature day to forecast next day with GroupBlend + confidence."""
    latest_day = df["date"].max()
    next_day = latest_day + pd.Timedelta(days=1)
    df_latest = df[df["date"].eq(latest_day)].copy().sort_values("h3_index")

    log.info("[ML] Feature day: %s → Prediction day: %s (%d H3 cells)", latest_day.date(), next_day.date(), len(df_latest))

    hotspot_score, group_scores = _predict_groupblend(df_latest, artifact, train_medians)

    out = df_latest[["h3_index"]].copy().reset_index(drop=True)
    out["prediction_date"] = next_day.date()
    out["hotspot_score"] = hotspot_score.astype(float)
    out["crime_probability"] = out["hotspot_score"]
    out["risk_rank"] = out["hotspot_score"].rank(method="first", ascending=False).astype(int)
    out["risk_rank_pct"] = out["risk_rank"] / max(len(out), 1)
    out["risk_level"] = assign_dashboard_risk_levels_by_rank(out["hotspot_score"]).values
    out["model_version"] = model_version_str()
    out["created_at"] = pd.Timestamp.utcnow().isoformat()
    out["pred_hotspot"] = (out["hotspot_score"].values >= best_threshold).astype(int)

    out = add_confidence_columns(out, group_scores)

    metric_map = {
        "val_auc": val_res.get("auc", np.nan), "val_ap": val_res.get("ap", np.nan), "val_f1": val_res.get("f1", np.nan),
        "test_auc": test_res.get("auc", np.nan), "test_ap": test_res.get("ap", np.nan), "test_f1": test_res.get("f1", np.nan),
        "val_top05_recall": val_res.get("top05_recall", np.nan), "val_top05_lift": val_res.get("top05_lift", np.nan),
        "val_top15_recall": val_res.get("top15_recall", np.nan), "val_top15_lift": val_res.get("top15_lift", np.nan),
        "val_top30_recall": val_res.get("top30_recall", np.nan), "val_top30_lift": val_res.get("top30_lift", np.nan),
        "test_top05_recall": test_res.get("top05_recall", np.nan), "test_top05_lift": test_res.get("top05_lift", np.nan),
        "test_top15_recall": test_res.get("top15_recall", np.nan), "test_top15_lift": test_res.get("top15_lift", np.nan),
        "test_top30_recall": test_res.get("top30_recall", np.nan), "test_top30_lift": test_res.get("top30_lift", np.nan),
    }
    for k, v in metric_map.items():
        out[k] = round(float(v), 4) if pd.notna(v) else np.nan

    ctx_cols = [
        "community_area", "hardship_index", "unemployment_rate", "population_density",
        "h3_train_percentile", "h3_train_active_rate", "roll_mean_7", "roll_mean_30", "zero_streak",
        "nbr_r1_mean", "nbr_r1_active_rate", "nbr_r1_active_count", "near_repeat_3d", "arrest_rate_7", "is_holiday",
    ]
    latest_reset = df_latest.reset_index(drop=True)
    for c in ctx_cols:
        if c in latest_reset.columns:
            out[c] = latest_reset[c].values

    core = [
        "prediction_date", "h3_index", "crime_probability", "hotspot_score",
        "risk_level", "confidence_level", "confidence_score", "model_version", "created_at",
    ]
    extra = [c for c in out.columns if c not in core]
    out = out[core + extra].sort_values("hotspot_score", ascending=False).reset_index(drop=True)

    log.info("[ML] Predictions – cells=%d | diagnostic pred_hotspot=%d (%.1f%%)", len(out), int(out["pred_hotspot"].sum()), out["pred_hotspot"].mean() * 100)
    log.info("[ML] Rank-bounded risk distribution:\n%s", out["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).fillna(0).astype(int).to_string())
    log.info("[ML] Confidence distribution:\n%s", out["confidence_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0).astype(int).to_string())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 8.  WRITE TO BIGQUERY
# ─────────────────────────────────────────────────────────────────────────────

def write_predictions_to_bq(
    client: bigquery.Client,
    df_pred: pd.DataFrame,
    project_id: str,
    dataset: str,
    predictions_table: str,
    write_mode: str = "WRITE_TRUNCATE",
) -> None:
    """
    Ghi DataFrame kết quả lên BigQuery.
    write_mode: WRITE_TRUNCATE (replace ngày hôm đó) hoặc WRITE_APPEND
    """
    dest = f"{project_id}.{dataset}.{predictions_table}"
    log.info("[ML] Writing %d predictions → %s (mode=%s)", len(df_pred), dest, write_mode)

    # Chỉ giữ các cột có trong schema
    schema_fields = {f.name for f in BQ_PREDICTION_SCHEMA}
    cols_to_write = [c for c in df_pred.columns if c in schema_fields]
    df_to_write   = df_pred[cols_to_write].copy()

    # Ép kiểu an toàn
    if "prediction_date" in df_to_write.columns:
        df_to_write["prediction_date"] = pd.to_datetime(df_to_write["prediction_date"]).dt.date
    if "created_at" in df_to_write.columns:
        df_to_write["created_at"] = pd.to_datetime(df_to_write["created_at"])

    job_config = bigquery.LoadJobConfig(
        write_disposition = write_mode,
        schema            = BQ_PREDICTION_SCHEMA,
    )
    job = client.load_table_from_dataframe(df_to_write, dest, job_config=job_config)
    job.result()
    log.info("[ML] ✓ Predictions written to BigQuery.")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  MAIN ENTRYPOINT (Airflow PythonOperator callable)
# ─────────────────────────────────────────────────────────────────────────────

def train_and_predict(
    project_id:        str,
    dataset:           str,
    enriched_table:    str  = "enriched_crime_data",
    predictions_table: str  = "prediction_results",
    lookback_days:     int  = RECENT_DAYS_TO_KEEP,
    model_output_path: str  = "/tmp/crime_model_v24_groupblend_confidence.pkl",
    write_mode:        str  = "WRITE_TRUNCATE",
    **kwargs,
) -> dict:
    """
    Airflow PythonOperator callable.

    BigData production version of the Kaggle Run 4 GroupBlend confidence ranker.
    Main target is next_hotspot = (next_total >= 2); output risk levels are
    bounded by top-k ranking to keep HIGH/CRITICAL controlled.
    """
    log.info("=" * 72)
    log.info("  Chicago Crime Hotspot Forecasting – ml_pipeline v24 GroupBlend confidence")
    log.info("  Target: next_hotspot = (next_total >= 2)")
    log.info("  Dashboard risk: CRITICAL=top5%%, HIGH+=top15%%, MEDIUM+=top30%%")
    log.info("=" * 72)

    client = bigquery.Client(project=project_id)
    df_raw = load_enriched_data(client, project_id, dataset, enriched_table, lookback_days)

    daily, all_h3 = aggregate_daily(df_raw)
    nbr_r1, nbr_r2, has_h3 = build_neighbor_maps(all_h3)

    # Rolling split dates computed before feature engineering, exactly as Kaggle v20.
    df_temp = daily.sort_values(["h3_index", "date"]).reset_index(drop=True)
    g_temp = df_temp.groupby("h3_index", group_keys=False)
    df_temp["next_total"] = g_temp["total_crimes"].shift(-1)
    supervised_dates = sorted(pd.to_datetime(df_temp.loc[df_temp["next_total"].notna(), "date"].unique()))
    eligible = supervised_dates[MIN_HISTORY_DAYS:]
    needed = TRAIN_DAYS + CALIB_DAYS + VAL_DAYS + TEST_DAYS
    if len(eligible) < needed:
        raise ValueError(f"Not enough history ({len(eligible)} eligible days). Need at least {needed} days.")

    test_dates  = eligible[-TEST_DAYS:]
    val_dates   = eligible[-(TEST_DAYS + VAL_DAYS):-TEST_DAYS]
    calib_dates = eligible[-(TEST_DAYS + VAL_DAYS + CALIB_DAYS):-(TEST_DAYS + VAL_DAYS)]
    train_end   = len(eligible) - (TEST_DAYS + VAL_DAYS + CALIB_DAYS)
    train_start = max(0, train_end - TRAIN_DAYS)
    train_dates = eligible[train_start:train_end]

    log.info("[ML] Split Train: %s→%s (%d d) Calib: %s→%s Val: %s→%s Test: %s→%s",
             train_dates[0].date(), train_dates[-1].date(), len(train_dates),
             calib_dates[0].date(), calib_dates[-1].date(),
             val_dates[0].date(), val_dates[-1].date(),
             test_dates[0].date(), test_dates[-1].date())

    df_feat = build_features(daily, all_h3, nbr_r1, nbr_r2, has_h3, train_dates)

    g_feat = df_feat.groupby("h3_index", group_keys=False)
    df_feat["next_total"] = g_feat["total_crimes"].shift(-1)
    df_feat["next_has_crime"] = (df_feat["next_total"] >= 1).astype(np.int8)
    df_feat["next_hotspot"] = (df_feat["next_total"] >= 2).astype(np.int8)
    df_feat["next_topk_group"] = np.nan
    supervised = df_feat["next_total"].notna()
    df_feat.loc[supervised & (df_feat["next_total"] <= 0), "next_topk_group"] = 0
    df_feat.loc[supervised & (df_feat["next_total"] == 1), "next_topk_group"] = 1
    df_feat.loc[supervised & (df_feat["next_total"] >= 2), "next_topk_group"] = 2
    df_feat["next_rank_pct"] = np.nan

    for name, dates in [("train", train_dates), ("calib", calib_dates), ("val", val_dates), ("test", test_dates)]:
        m = df_feat["date"].isin(dates) & supervised
        log.info("[ML] %s next_has_crime=%.2f%% | next_hotspot>=2=%.2f%%", name, df_feat.loc[m, "next_has_crime"].mean()*100, df_feat.loc[m, "next_hotspot"].mean()*100)

    artifact, feature_cols, train_medians, best_threshold, val_res, test_res, sample_weight_info = train_model(
        df_feat, train_dates, calib_dates, val_dates, test_dates
    )

    df_pred = build_predictions(df_feat, artifact, train_medians, best_threshold, val_res, test_res)
    write_predictions_to_bq(client, df_pred, project_id, dataset, predictions_table, write_mode)

    meta = {
        "version": model_version_str(),
        "target": "next_hotspot = next_total >= 2",
        "risk_policy": "CRITICAL top5%, HIGH cumulative top15%, MEDIUM cumulative top30%",
        "feature_cols": feature_cols,
        "train_medians": train_medians,
        "best_threshold_diagnostic": best_threshold,
        "group_weights": GROUP_BLEND_WEIGHTS,
        "sample_weight_info": sample_weight_info,
        "group_weights": GROUP_BLEND_WEIGHTS,
        "train_dates": train_dates,
        "calib_dates": calib_dates,
        "val_dates": val_dates,
        "test_dates": test_dates,
        "model_type": "GroupBlend LightGBM temporal/spatial/context",
        "val_res": val_res,
        "test_res": test_res,
        "has_h3_spillover": has_h3,
        "artifact": artifact,
    }
    os.makedirs(os.path.dirname(model_output_path) or ".", exist_ok=True)
    with open(model_output_path, "wb") as f:
        pickle.dump({"artifact": artifact, "meta": meta}, f)
    log.info("[ML] Model artifact saved → %s", model_output_path)

    risk_counts = df_pred["risk_level"].value_counts().to_dict()
    log.info("=" * 72)
    log.info("  SUMMARY v24 GroupBlend confidence")
    log.info("  VAL  AUC=%.4f AP=%.4f F1=%.4f Top05_R=%.4f Top05_L=%.2fx Top15_R=%.4f Top30_R=%.4f",
             val_res["auc"], val_res["ap"], val_res["f1"], val_res["top05_recall"], val_res["top05_lift"], val_res["top15_recall"], val_res["top30_recall"])
    log.info("  TEST AUC=%.4f AP=%.4f F1=%.4f Top05_R=%.4f Top05_L=%.2fx Top15_R=%.4f Top30_R=%.4f",
             test_res["auc"], test_res["ap"], test_res["f1"], test_res["top05_recall"], test_res["top05_lift"], test_res["top15_recall"], test_res["top30_recall"])
    log.info("  Risk counts: %s", risk_counts)
    log.info("=" * 72)

    return {
        "val_auc": val_res["auc"], "val_ap": val_res["ap"], "val_f1": val_res["f1"],
        "test_auc": test_res["auc"], "test_ap": test_res["ap"], "test_f1": test_res["f1"],
        "val_top05_recall": val_res["top05_recall"], "val_top05_lift": val_res["top05_lift"],
        "test_top05_recall": test_res["top05_recall"], "test_top05_lift": test_res["top05_lift"],
        "threshold_diagnostic": best_threshold,
        "group_weights": GROUP_BLEND_WEIGHTS,
        "risk_counts": risk_counts,
        "n_predictions": len(df_pred),
        "model_version": model_version_str(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 10. AIRFLOW DAG HELPER (gợi ý dùng trong DAG file)
# ─────────────────────────────────────────────────────────────────────────────

def make_airflow_callable(
    project_id: str,
    dataset:    str,
    **kwargs,
):
    """
    Trả về callable đã bind sẵn config, để dùng trong PythonOperator:

    Example DAG usage:
    ------------------
    from ml_pipeline import make_airflow_callable

    run_ml = PythonOperator(
        task_id="train_predict_v24_groupblend",
        python_callable=make_airflow_callable(
            project_id="sage-mind-489618-n5",
            dataset="cdash_warehouse",
        ),
        dag=dag,
    )
    """
    def _callable(**airflow_kwargs):
        return train_and_predict(
            project_id        = project_id,
            dataset           = dataset,
            enriched_table    = kwargs.get("enriched_table",    "enriched_crime_data"),
            predictions_table = kwargs.get("predictions_table", "prediction_results"),
            lookback_days     = kwargs.get("lookback_days",     RECENT_DAYS_TO_KEEP),
            model_output_path = kwargs.get("model_output_path", "/tmp/crime_model_v24_groupblend_confidence.pkl"),
            write_mode        = kwargs.get("write_mode",        "WRITE_TRUNCATE"),
        )
    return _callable


# ─────────────────────────────────────────────────────────────────────────────
# CLI – chạy thủ công để test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s %(levelname)s %(message)s",
        datefmt = "%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Crime Hotspot ML Pipeline v20 target-safe")
    parser.add_argument("--project",    required=True,  help="GCP Project ID")
    parser.add_argument("--dataset",    required=True,  help="BigQuery dataset")
    parser.add_argument("--enriched",   default="enriched_crime_data")
    parser.add_argument("--predictions",default="prediction_results")
    parser.add_argument("--lookback",   type=int, default=RECENT_DAYS_TO_KEEP)
    parser.add_argument("--model_path", default="/tmp/crime_model_v24_groupblend_confidence.pkl")
    parser.add_argument("--write_mode", default="WRITE_TRUNCATE",
                        choices=["WRITE_TRUNCATE", "WRITE_APPEND"])
    args = parser.parse_args()

    result = train_and_predict(
        project_id        = args.project,
        dataset           = args.dataset,
        enriched_table    = args.enriched,
        predictions_table = args.predictions,
        lookback_days     = args.lookback,
        model_output_path = args.model_path,
        write_mode        = args.write_mode,
    )
    print("\n✅ Done:", result)
    sys.exit(0)