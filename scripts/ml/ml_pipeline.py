"""
ml_pipeline.py  –  Crime Hotspot Forecasting v11
=================================================
Airflow PythonOperator callable.
Đọc enriched_crime_data từ BigQuery → Feature Engineering v11 →
Train XGBoost → Ghi prediction_results về BigQuery.

Cải tiến v11 (so với v1):
  [4G] H3 Spatial Spillover          – Routine Activity Theory
  [4I] Near-Repeat Victimization     – Near-Repeat Theory
  [4J] Cross-Crime Lead-Lag          – Escalation Theory
  [4K] Arrest Deterrence             – Deterrence Theory
  [4L] Density-Normalized Crime      – Density Bias Correction
  [4M] Holiday / Seasonal Events     – Routine Activity Theory
  [4N] Momentum + Trend Direction    – Temporal Momentum
  [4O] Hardship Interaction          – Social Disorganization Theory

Output schema (prediction_results):
  prediction_date | h3_index | crime_probability | risk_level |
  model_version   | created_at
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
log = logging.getLogger("ML_Pipeline_v11")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONSTANTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL_VERSION_PREFIX = "crime_prob_v11"

# Rolling split windows (ngày)
TRAIN_DAYS       = 180
CALIB_DAYS       = 14
VAL_DAYS         = 7
TEST_DAYS        = 7
MIN_HISTORY_DAYS = 60

RANDOM_STATE = 42

# XGBoost hyperparameters (đã tune cho bài toán binary hotspot)
XGB_PARAMS = dict(
    n_estimators          = 3000,
    learning_rate         = 0.018,
    max_depth             = 5,
    min_child_weight      = 10,
    subsample             = 0.85,
    colsample_bytree      = 0.65,
    colsample_bylevel     = 0.75,
    colsample_bynode      = 0.85,
    reg_alpha             = 0.40,
    reg_lambda            = 5.0,
    objective             = "binary:logistic",
    eval_metric           = "aucpr",
    tree_method           = "hist",
    random_state          = RANDOM_STATE,
    n_jobs                = -1,
    early_stopping_rounds = 120,
)

THRESHOLD_GRID = np.arange(0.05, 0.65, 0.01).tolist()

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
    bigquery.SchemaField("crime_probability", "FLOAT"),
    bigquery.SchemaField("risk_level",        "STRING"),
    bigquery.SchemaField("model_version",     "STRING"),
    bigquery.SchemaField("created_at",        "TIMESTAMP"),
    # Extended debug / context columns
    bigquery.SchemaField("pred_has_crime",    "INTEGER"),
    bigquery.SchemaField("risk_rank",         "INTEGER"),
    bigquery.SchemaField("val_auc",           "FLOAT"),
    bigquery.SchemaField("val_f1",            "FLOAT"),
    bigquery.SchemaField("test_auc",          "FLOAT"),
    bigquery.SchemaField("test_f1",           "FLOAT"),
    bigquery.SchemaField("community_area",    "FLOAT"),
    bigquery.SchemaField("hardship_index",    "FLOAT"),
    bigquery.SchemaField("unemployment_rate", "FLOAT"),
    bigquery.SchemaField("population_density","FLOAT"),
    bigquery.SchemaField("roll_mean_7",       "FLOAT"),
    bigquery.SchemaField("roll_mean_30",      "FLOAT"),
    bigquery.SchemaField("zero_streak",       "FLOAT"),
    bigquery.SchemaField("nbr_r1_mean",       "FLOAT"),
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
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full_like(num, default, dtype=float)
    mask = np.abs(den) > 1e-12
    out[mask] = num[mask] / den[mask]
    if clip is not None:
        out = np.clip(out, -clip, clip)
    return out


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
    x = df.copy()
    for c in cols:
        if c not in x.columns:
            x[c] = 0.0
        x[c] = pd.to_numeric(x[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = x[cols].median(numeric_only=True).fillna(0.0)
    x[cols] = x[cols].fillna(medians).fillna(0.0)
    return x[cols], medians


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
    return pd.cut(
        prob,
        bins=[0.0, 0.30, 0.60, 0.80, 1.01],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        right=False,
        include_lowest=True,
    ).astype(str)


def model_version_str() -> str:
    return pd.Timestamp.now().strftime("%Y-%m-%d") + f".{MODEL_VERSION_PREFIX}"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOADING FROM BIGQUERY
# ─────────────────────────────────────────────────────────────────────────────

def load_enriched_data(
    client: bigquery.Client,
    project_id: str,
    dataset: str,
    enriched_table: str,
    lookback_days: int = 400,
) -> pd.DataFrame:
    """
    Kéo enriched_crime_data từ BigQuery.
    Chỉ lấy ~lookback_days ngày gần nhất để tránh load toàn bộ lịch sử.
    """
    log.info("[ML] Loading enriched_crime_data from BigQuery (last %d days)...", lookback_days)

    query = f"""
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
                 THEN 1 ELSE 0 END              AS is_evening
        FROM `{project_id}.{dataset}.{enriched_table}`
        WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL {lookback_days} DAY)
          AND h3_index IS NOT NULL
          AND latitude  IS NOT NULL
          AND longitude IS NOT NULL
    """
    df = client.query(query).to_dataframe()
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

    log.info("[ML] Daily grid: %s rows, %d H3 cells, %d dates",
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
            disk1 = set(h3lib.grid_disk(cell, 1))
            disk2 = set(h3lib.grid_disk(cell, 2))
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
    Feature engineering v11 – RAM-optimised rewrite.

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
    # RAM-OPTIMISED: dùng float32, tính từng loại nbr rồi xoá ngay,
    # không giữ 5 ma trận (n_h3 × n_dates) cùng lúc trong bộ nhớ.
    log.info("[ML]   [4G] H3 spatial spillover (Routine Activity Theory)")
    if has_h3:
        # --- Pivot với float32 để tiết kiệm 50% RAM so với float64 ---
        pivot_src = daily[["date", "h3_index", "total_crimes", "crime_today"]].copy()
        pivot_src["total_crimes"] = pivot_src["total_crimes"].astype("float32")
        pivot_src["crime_today"]  = pivot_src["crime_today"].astype("float32")

        crime_pivot  = pivot_src.pivot_table(
            index="date", columns="h3_index", values="total_crimes", fill_value=0
        ).astype("float32")
        active_pivot = pivot_src.pivot_table(
            index="date", columns="h3_index", values="crime_today", fill_value=0
        ).astype("float32")
        del pivot_src
        gc.collect()

        h3_cols    = crime_pivot.columns.tolist()
        date_index = crime_pivot.index

        def _build_nbr_matrix(src_pivot: pd.DataFrame, nbr_map: dict, agg: str) -> pd.DataFrame:
            """
            Tính ma trận neighbour (n_dates × n_h3) theo agg = mean/sum/max.
            Dùng numpy để tránh overhead của pandas, trả về float32.
            """
            h3_pos = {cell: i for i, cell in enumerate(h3_cols)}
            src_np = src_pivot.values  # shape: (n_dates, n_h3), float32

            out_np = np.zeros_like(src_np, dtype="float32")
            for i, cell in enumerate(h3_cols):
                nbrs = [h3_pos[n] for n in nbr_map.get(cell, []) if n in h3_pos]
                if not nbrs:
                    continue
                nbr_vals = src_np[:, nbrs]   # (n_dates, k)
                if agg == "mean":
                    out_np[:, i] = nbr_vals.mean(axis=1)
                elif agg == "sum":
                    out_np[:, i] = nbr_vals.sum(axis=1)
                elif agg == "max":
                    out_np[:, i] = nbr_vals.max(axis=1)

            return pd.DataFrame(out_np, index=date_index, columns=h3_cols)

        def _lag_melt(piv: pd.DataFrame, col_name: str) -> pd.DataFrame:
            """shift(1) → melt → float32, xoá pivot ngay sau khi dùng."""
            melted = (
                piv.shift(1)
                .reset_index()
                .melt(id_vars="date", var_name="h3_index", value_name=col_name)
            )
            melted[col_name] = melted[col_name].astype("float32")
            return melted

        # Tính từng loại nbr, melt ngay, xoá ma trận lớn trước khi tính cái tiếp theo
        log.info("[ML]     [4G] computing nbr_r1_mean ...")
        nbr_r1_mean_d = _build_nbr_matrix(crime_pivot, nbr_r1, "mean")
        nbr_merged    = _lag_melt(nbr_r1_mean_d, "nbr_r1_mean")

        # nbr_r1_roll7 cần nbr_r1_mean_d → tính luôn trước khi xoá
        log.info("[ML]     [4G] computing nbr_r1_roll7 ...")
        nbr_roll7_melt = (
            nbr_r1_mean_d.shift(1).rolling(7, min_periods=2).mean()
            .reset_index()
            .melt(id_vars="date", var_name="h3_index", value_name="nbr_r1_roll7")
        )
        nbr_roll7_melt["nbr_r1_roll7"] = nbr_roll7_melt["nbr_r1_roll7"].astype("float32")
        del nbr_r1_mean_d
        gc.collect()

        log.info("[ML]     [4G] computing nbr_r1_sum ...")
        nbr_r1_sum_d = _build_nbr_matrix(crime_pivot, nbr_r1, "sum")
        nd = _lag_melt(nbr_r1_sum_d, "nbr_r1_sum")
        nbr_merged = nbr_merged.merge(nd, on=["date", "h3_index"], how="left")
        del nbr_r1_sum_d, nd
        gc.collect()

        log.info("[ML]     [4G] computing nbr_r1_max ...")
        nbr_r1_max_d = _build_nbr_matrix(crime_pivot, nbr_r1, "max")
        nd = _lag_melt(nbr_r1_max_d, "nbr_r1_max")
        nbr_merged = nbr_merged.merge(nd, on=["date", "h3_index"], how="left")
        del nbr_r1_max_d, nd
        gc.collect()

        log.info("[ML]     [4G] computing nbr_r1_active_rate ...")
        nbr_r1_active_d = _build_nbr_matrix(active_pivot, nbr_r1, "mean")
        nd = _lag_melt(nbr_r1_active_d, "nbr_r1_active_rate")
        nbr_merged = nbr_merged.merge(nd, on=["date", "h3_index"], how="left")
        del nbr_r1_active_d, nd, active_pivot
        gc.collect()

        log.info("[ML]     [4G] computing nbr_r2_mean ...")
        nbr_r2_mean_d = _build_nbr_matrix(crime_pivot, nbr_r2, "mean")
        nd = _lag_melt(nbr_r2_mean_d, "nbr_r2_mean")
        nbr_merged = nbr_merged.merge(nd, on=["date", "h3_index"], how="left")
        del nbr_r2_mean_d, nd, crime_pivot
        gc.collect()

        # Gắn nbr_roll7
        nbr_merged = nbr_merged.merge(nbr_roll7_melt, on=["date", "h3_index"], how="left")
        del nbr_roll7_melt
        gc.collect()

        # Merge vào df chính
        df = df.merge(nbr_merged, on=["date", "h3_index"], how="left")
        del nbr_merged
        gc.collect()

        for c in ["nbr_r1_mean", "nbr_r1_sum", "nbr_r1_max",
                  "nbr_r1_active_rate", "nbr_r2_mean", "nbr_r1_roll7"]:
            df[c] = df[c].fillna(0).astype("float32")

        df["h3_vs_nbr_r1"]  = safe_ratio(
            df["roll_mean_7"], df["nbr_r1_mean"] + 1e-3, default=0, clip=20
        ).astype("float32")
        df["h3_above_nbr"]  = (df["lag_1"] > df["nbr_r1_mean"]).astype("int8")
        df["nbr_contagion"] = (df["nbr_r1_active_rate"] > 0.5).astype("int8")

        log.info("[ML]     ✓ Spatial spillover features added")
    else:
        for c in ["nbr_r1_mean", "nbr_r1_sum", "nbr_r1_max",
                  "nbr_r1_active_rate", "nbr_r2_mean", "nbr_r1_roll7",
                  "h3_vs_nbr_r1", "h3_above_nbr", "nbr_contagion"]:
            df[c] = np.float32(0)
        log.warning("[ML]     ⚠️  Spatial spillover skipped (h3 not installed)")

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
    h3_prior["h3_train_active_rate"] = (
        df.loc[train_mask_prior]
        .groupby("h3_index")["crime_today"]
        .mean()
        .reindex(h3_prior["h3_index"])
        .values
    )
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
    Train XGBoost, calibrate threshold, evaluate.
    Returns: (model, feature_cols, train_medians, best_threshold, val_res, test_res)
    """
    import xgboost as xgb
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        precision_score, recall_score, f1_score,
    )

    log.info("[ML] Preparing train/calib/val/test splits...")

    ID_COLS     = {"h3_index", "date"}
    TARGET_COLS = {"next_total", "next_has_crime"}
    exclude     = ID_COLS | TARGET_COLS

    feature_cols = [
        c for c in df.columns
        if c not in exclude
        and not c.startswith("next_")
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    supervised = df["next_total"].notna()
    train_mask = df["date"].isin(train_dates) & supervised
    calib_mask = df["date"].isin(calib_dates) & supervised
    val_mask   = df["date"].isin(val_dates)   & supervised
    test_mask  = df["date"].isin(test_dates)  & supervised

    # Drop constant / all-null features
    keep = []
    for c in feature_cols:
        s = pd.to_numeric(df.loc[train_mask, c], errors="coerce")
        if s.notna().sum() > 0 and s.nunique(dropna=True) > 1:
            keep.append(c)
    feature_cols = keep

    X_train, train_medians = sanitize_numeric(df.loc[train_mask], feature_cols)
    X_calib, _             = sanitize_numeric(df.loc[calib_mask], feature_cols, train_medians)
    X_val,   _             = sanitize_numeric(df.loc[val_mask],   feature_cols, train_medians)
    X_test,  _             = sanitize_numeric(df.loc[test_mask],  feature_cols, train_medians)

    y_train = df.loc[train_mask, "next_has_crime"].astype(int)
    y_calib = df.loc[calib_mask, "next_has_crime"].astype(int)
    y_val   = df.loc[val_mask,   "next_has_crime"].astype(int)
    y_test  = df.loc[test_mask,  "next_has_crime"].astype(int)

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)

    near_repeat = (
        df.loc[train_mask, "near_repeat_3d"].fillna(0).values
        if "near_repeat_3d" in df.columns else 0
    )
    next_total_train = df.loc[train_mask, "next_total"].astype(float).clip(0, 5).values
    sample_weight = np.where(
        y_train == 1,
        1.0 + 0.3 * next_total_train + 0.2 * near_repeat,
        1.0,
    )

    log.info("[ML] Train=%d  Calib=%d  Val=%d  Test=%d  features=%d",
             len(X_train), len(X_calib), len(X_val), len(X_test), len(feature_cols))
    log.info("[ML] Positive rate train=%.2f%%  scale_pos_weight=%.2f",
             y_train.mean() * 100, scale_pos_weight)

    params = dict(XGB_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_train, y_train), (X_calib, y_calib)],
        verbose=200,
    )
    log.info("[ML] Training done. Best iteration: %s", getattr(model, "best_iteration", "n/a"))

    calib_prob = model.predict_proba(X_calib)[:, 1]
    val_prob   = model.predict_proba(X_val)[:, 1]
    test_prob  = model.predict_proba(X_test)[:, 1]

    best_t, _ = find_best_threshold(y_calib.values, calib_prob)
    val_pred   = (val_prob  >= best_t).astype(int)
    test_pred  = (test_prob >= best_t).astype(int)

    def _metrics(y_true, y_pred, y_prob):
        return {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall":    float(recall_score(y_true,    y_pred, zero_division=0)),
            "f1":        float(f1_score(y_true,        y_pred, zero_division=0)),
            "auc":       float(roc_auc_score(y_true, y_prob)),
            "ap":        float(average_precision_score(y_true, y_prob)),
        }

    val_res  = _metrics(y_val.values,  val_pred,  val_prob)
    test_res = _metrics(y_test.values, test_pred, test_prob)

    log.info(
        "[ML] VAL  – P=%.4f R=%.4f F1=%.4f AUC=%.4f AP=%.4f",
        val_res["precision"], val_res["recall"], val_res["f1"],
        val_res["auc"],       val_res["ap"],
    )
    log.info(
        "[ML] TEST – P=%.4f R=%.4f F1=%.4f AUC=%.4f AP=%.4f",
        test_res["precision"], test_res["recall"], test_res["f1"],
        test_res["auc"],       test_res["ap"],
    )
    log.info("[ML] Calibrated threshold (F1-optimal on CALIB): %.2f", best_t)

    return model, feature_cols, train_medians, best_t, val_res, test_res, scale_pos_weight


# ─────────────────────────────────────────────────────────────────────────────
# 7.  INFERENCE: NEXT DAY PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

def build_predictions(
    df: pd.DataFrame,
    model,
    feature_cols: list[str],
    train_medians: pd.Series,
    best_threshold: float,
    val_res: dict,
    test_res: dict,
) -> pd.DataFrame:
    """
    Dùng features từ ngày mới nhất → dự báo ngày tiếp theo.
    Output đúng schema prediction_results.
    """
    latest_day = df["date"].max()
    next_day   = latest_day + pd.Timedelta(days=1)
    df_latest  = df[df["date"].eq(latest_day)].copy().sort_values("h3_index")

    log.info("[ML] Feature day: %s  →  Prediction day: %s  (%d H3 cells)",
             latest_day.date(), next_day.date(), len(df_latest))

    X_latest, _ = sanitize_numeric(df_latest, feature_cols, train_medians)
    crime_prob  = model.predict_proba(X_latest)[:, 1]

    out = df_latest[["h3_index"]].copy().reset_index(drop=True)
    out["prediction_date"]   = next_day.date()
    out["crime_probability"] = crime_prob
    out["risk_level"]        = risk_level_from_prob(pd.Series(crime_prob)).values
    out["model_version"]     = model_version_str()
    out["created_at"]        = pd.Timestamp.utcnow().isoformat()

    out["pred_has_crime"] = (crime_prob >= best_threshold).astype(int)
    out["risk_rank"]      = (
        pd.Series(crime_prob).rank(method="first", ascending=False).astype(int).values
    )
    out["val_auc"]  = round(val_res["auc"],  4)
    out["val_f1"]   = round(val_res["f1"],   4)
    out["test_auc"] = round(test_res["auc"], 4)
    out["test_f1"]  = round(test_res["f1"],  4)

    ctx_cols = [
        "community_area", "hardship_index", "unemployment_rate", "population_density",
        "h3_train_percentile", "roll_mean_7", "roll_mean_30", "zero_streak",
        "nbr_r1_mean", "near_repeat_3d", "arrest_rate_7", "is_holiday",
    ]
    for c in ctx_cols:
        if c in df_latest.columns:
            out[c] = df_latest[c].values

    core = [
        "prediction_date", "h3_index", "crime_probability",
        "risk_level", "model_version", "created_at",
    ]
    extra = [c for c in out.columns if c not in core]
    out   = out[core + extra].sort_values("crime_probability", ascending=False)

    log.info("[ML] Predictions – cells=%d  predicted_crime=%d (%.1f%%)",
             len(out),
             out["pred_has_crime"].sum(),
             out["pred_has_crime"].mean() * 100)
    log.info("[ML] Risk distribution:\n%s",
             out["risk_level"].value_counts()
             .reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).fillna(0).astype(int).to_string())
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
    lookback_days:     int  = 400,
    model_output_path: str  = "/tmp/crime_model_v11.pkl",
    write_mode:        str  = "WRITE_TRUNCATE",
    **kwargs,
) -> dict:
    """
    Airflow PythonOperator callable.

    Parameters
    ----------
    project_id        : GCP project ID
    dataset           : BigQuery dataset name (e.g. 'cdash_warehouse')
    enriched_table    : Source table (default 'enriched_crime_data')
    predictions_table : Destination table (default 'prediction_results')
    lookback_days     : How many days of history to pull (default 400)
    model_output_path : Where to pickle the model for auditing
    write_mode        : 'WRITE_TRUNCATE' (default) hoặc 'WRITE_APPEND'

    Returns
    -------
    dict với val_auc, test_auc, val_f1, test_f1, n_predictions, threshold
    """
    log.info("=" * 72)
    log.info("  Chicago Crime Hotspot Forecasting  –  ml_pipeline v11")
    log.info("  Target: next_has_crime = (next_total >= 1) | schema OK")
    log.info("=" * 72)

    # ── Step 1: Load data ─────────────────────────────────────────────────
    client  = bigquery.Client(project=project_id)
    df_raw  = load_enriched_data(
        client, project_id, dataset, enriched_table, lookback_days
    )

    # ── Step 2: Aggregate daily per H3 ───────────────────────────────────
    daily, all_h3 = aggregate_daily(df_raw)

    # ── Step 3: Build neighbor maps ───────────────────────────────────────
    nbr_r1, nbr_r2, has_h3 = build_neighbor_maps(all_h3)

    # ── Step 4: Define rolling split dates ───────────────────────────────
    df_temp   = daily.sort_values(["h3_index", "date"]).reset_index(drop=True)
    g_temp    = df_temp.groupby("h3_index", group_keys=False)
    df_temp["next_total"] = g_temp["total_crimes"].shift(-1)
    supervised_dates = sorted(
        pd.to_datetime(df_temp.loc[df_temp["next_total"].notna(), "date"].unique())
    )
    eligible = supervised_dates[MIN_HISTORY_DAYS:]

    if len(eligible) < TEST_DAYS + VAL_DAYS + CALIB_DAYS + 30:
        raise ValueError(
            f"Not enough history ({len(eligible)} eligible days). "
            f"Need at least {TEST_DAYS + VAL_DAYS + CALIB_DAYS + 30} days."
        )

    test_dates  = eligible[-TEST_DAYS:]
    val_dates   = eligible[-(TEST_DAYS + VAL_DAYS):-TEST_DAYS]
    calib_dates = eligible[-(TEST_DAYS + VAL_DAYS + CALIB_DAYS):-(TEST_DAYS + VAL_DAYS)]
    train_end   = len(eligible) - (TEST_DAYS + VAL_DAYS + CALIB_DAYS)
    train_start = max(0, train_end - TRAIN_DAYS)
    train_dates = eligible[train_start:train_end]

    log.info("[ML] Split  Train: %s→%s (%d d)  Val: %s→%s  Test: %s→%s",
             train_dates[0].date(), train_dates[-1].date(), len(train_dates),
             val_dates[0].date(),   val_dates[-1].date(),
             test_dates[0].date(),  test_dates[-1].date())

    # ── Step 5: Feature engineering v11 ──────────────────────────────────
    df_feat = build_features(daily, all_h3, nbr_r1, nbr_r2, has_h3, train_dates)

    # Add target
    g_feat = df_feat.groupby("h3_index", group_keys=False)
    df_feat["next_total"]     = g_feat["total_crimes"].shift(-1)
    df_feat["next_has_crime"] = (df_feat["next_total"] >= 1).astype(np.int8)

    # ── Step 6: Train model ───────────────────────────────────────────────
    (
        model, feature_cols, train_medians,
        best_threshold, val_res, test_res, scale_pos_weight,
    ) = train_model(df_feat, train_dates, calib_dates, val_dates, test_dates)

    # ── Step 7: Build next-day predictions ───────────────────────────────
    df_pred = build_predictions(
        df_feat, model, feature_cols, train_medians,
        best_threshold, val_res, test_res,
    )

    # ── Step 8: Write to BigQuery ─────────────────────────────────────────
    write_predictions_to_bq(client, df_pred, project_id, dataset, predictions_table, write_mode)

    # ── Step 9: Persist model artifact ───────────────────────────────────
    meta = {
        "version":          model_version_str(),
        "feature_cols":     feature_cols,
        "train_medians":    train_medians,
        "best_threshold":   best_threshold,
        "scale_pos_weight": scale_pos_weight,
        "train_dates":      train_dates,
        "calib_dates":      calib_dates,
        "val_dates":        val_dates,
        "test_dates":       test_dates,
        "xgb_params":       XGB_PARAMS,
        "val_res":          val_res,
        "test_res":         test_res,
        "has_h3_spillover": has_h3,
        "neighbor_map_r1":  nbr_r1 if has_h3 else {},
        "neighbor_map_r2":  nbr_r2 if has_h3 else {},
    }
    os.makedirs(os.path.dirname(model_output_path) or ".", exist_ok=True)
    with open(model_output_path, "wb") as f:
        pickle.dump({"model": model, "meta": meta}, f)
    log.info("[ML] Model artifact saved → %s", model_output_path)

    # ── Summary ───────────────────────────────────────────────────────────
    log.info("=" * 72)
    log.info("  SUMMARY v11")
    log.info("  VAL  AUC=%.4f  F1=%.4f  AP=%.4f", val_res["auc"],  val_res["f1"],  val_res["ap"])
    log.info("  TEST AUC=%.4f  F1=%.4f  AP=%.4f", test_res["auc"], test_res["f1"], test_res["ap"])
    log.info("  Threshold (F1-calib)=%.2f  |  Predictions=%d", best_threshold, len(df_pred))
    log.info("=" * 72)

    return {
        "val_auc":      val_res["auc"],
        "val_f1":       val_res["f1"],
        "test_auc":     test_res["auc"],
        "test_f1":      test_res["f1"],
        "threshold":    best_threshold,
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
        task_id="train_predict_v11",
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
            lookback_days     = kwargs.get("lookback_days",     400),
            model_output_path = kwargs.get("model_output_path", "/tmp/crime_model_v11.pkl"),
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

    parser = argparse.ArgumentParser(description="Crime Hotspot ML Pipeline v11")
    parser.add_argument("--project",    required=True,  help="GCP Project ID")
    parser.add_argument("--dataset",    required=True,  help="BigQuery dataset")
    parser.add_argument("--enriched",   default="enriched_crime_data")
    parser.add_argument("--predictions",default="prediction_results")
    parser.add_argument("--lookback",   type=int, default=400)
    parser.add_argument("--model_path", default="/tmp/crime_model_v11.pkl")
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