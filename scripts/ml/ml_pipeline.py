"""
ml_pipeline.py
==============
Train XGBoost để dự đoán xác suất tội phạm theo H3 cell.
Được gọi bởi PythonOperator trong DAG.

Đặt file này trong: dags/ml_pipeline.py
(cùng thư mục với DAG để Airflow import được)
"""

import pandas as pd
import numpy as np
from google.cloud import bigquery


# ─────────────────────────────────────────────────────────────────────────────
# Feature columns dùng để train
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "hour_of_day",
    "day_of_week",
    "month",
    "temp_max",
    "temp_min",
    "precipitation",
    "wind_speed",
    "per_capita_income",
    "unemployment_rate",
    "hardship_index",
]

TARGET_COL = "is_violent"


def train_and_predict(project_id: str, dataset: str, enriched_table: str, predictions_table: str):
    """
    1. Kéo enriched_crime_data từ BigQuery
    2. Train XGBoost classifier
    3. Tính xác suất tội phạm cho tuần tới theo H3 cell
    4. Ghi kết quả vào prediction_results
    """
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    client = bigquery.Client(project=project_id)

    # ── 1. Load training data ─────────────────────────────────────────────────
    print("[ML] Loading enriched_crime_data from BigQuery...")
    query = f"""
        SELECT
            h3_index,
            {', '.join(FEATURE_COLS)},
            {TARGET_COL}
        FROM `{project_id}.{dataset}.{enriched_table}`
        WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR)
          AND h3_index IS NOT NULL
    """
    df = client.query(query).to_dataframe()
    print(f"[ML] Loaded {len(df):,} rows")

    # ── 2. Prep features ──────────────────────────────────────────────────────
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    X = df[FEATURE_COLS].astype(float)
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 3. Train XGBoost ──────────────────────────────────────────────────────
    print("[ML] Training XGBoost...")

    # scale_pos_weight xử lý class imbalance (ít violent hơn non-violent)
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos if pos > 0 else 1.0
    print(f"[ML] Class ratio neg/pos = {scale:.2f} → scale_pos_weight={scale:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"[ML] ✓ AUC-ROC on test set: {auc:.4f}")

    # Feature importance log
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    print("[ML] Feature importances:")
    print(importances.sort_values(ascending=False).to_string())

    # ── 5. Generate predictions cho tuần tới ──────────────────────────────────
    # Lấy distinct H3 cells + thống kê trung bình feature từ 30 ngày gần nhất
    print("[ML] Generating next-week predictions per H3 cell...")
    query_h3 = f"""
        SELECT
            h3_index,
            AVG(temp_max)           AS temp_max,
            AVG(temp_min)           AS temp_min,
            AVG(precipitation)      AS precipitation,
            AVG(wind_speed)         AS wind_speed,
            AVG(per_capita_income)  AS per_capita_income,
            AVG(unemployment_rate)  AS unemployment_rate,
            AVG(hardship_index)     AS hardship_index,
            COUNT(*)                AS recent_crime_count
        FROM `{project_id}.{dataset}.{enriched_table}`
        WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
          AND h3_index IS NOT NULL
        GROUP BY h3_index
    """
    df_h3 = client.query(query_h3).to_dataframe()
    print(f"[ML] Found {len(df_h3):,} distinct H3 cells")

    # Tạo predictions cho từng (h3_cell × hour_of_day × day_of_week)
    # Để đơn giản: dùng peak hours (18-22) và weekdays
    rows = []
    for _, row in df_h3.iterrows():
        for hour in [8, 12, 18, 20, 22]:       # các giờ cao điểm
            for dow in range(1, 8):             # thứ 2 → CN
                feat = {
                    "h3_index": row["h3_index"],
                    "hour_of_day": hour,
                    "day_of_week": dow,
                    "month": pd.Timestamp.now().month,
                    "temp_max": row["temp_max"],
                    "temp_min": row["temp_min"],
                    "precipitation": row["precipitation"],
                    "wind_speed": row["wind_speed"],
                    "per_capita_income": row["per_capita_income"],
                    "unemployment_rate": row["unemployment_rate"],
                    "hardship_index": row["hardship_index"],
                }
                rows.append(feat)

    df_pred_input = pd.DataFrame(rows)
    X_pred = df_pred_input[FEATURE_COLS].astype(float).fillna(0)
    df_pred_input["crime_probability"] = model.predict_proba(X_pred)[:, 1]

    # Aggregate: lấy max probability per H3 cell (worst case trong tuần)
    df_agg = (
        df_pred_input
        .groupby("h3_index")["crime_probability"]
        .max()
        .reset_index()
        .rename(columns={"crime_probability": "max_crime_probability"})
    )
    df_agg = df_agg.merge(
        df_h3[["h3_index", "recent_crime_count"]],
        on="h3_index", how="left"
    )
    df_agg["risk_level"] = pd.cut(
        df_agg["max_crime_probability"],
        bins=[0, 0.3, 0.6, 0.8, 1.0],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    ).astype(str)
    df_agg["prediction_date"] = pd.Timestamp.now().date().isoformat()
    df_agg["model_version"] = "xgb_v1"
    df_agg["auc_score"] = round(auc, 4)

    # ── 6. Ghi vào BigQuery ───────────────────────────────────────────────────
    dest_table = f"{project_id}.{dataset}.{predictions_table}"
    print(f"[ML] Writing {len(df_agg):,} predictions to {dest_table}")

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",  # overwrite mỗi tuần
        schema=[
            bigquery.SchemaField("h3_index",              "STRING"),
            bigquery.SchemaField("max_crime_probability", "FLOAT"),
            bigquery.SchemaField("recent_crime_count",    "INTEGER"),
            bigquery.SchemaField("risk_level",            "STRING"),
            bigquery.SchemaField("prediction_date",       "DATE"),
            bigquery.SchemaField("model_version",         "STRING"),
            bigquery.SchemaField("auc_score",             "FLOAT"),
        ],
    )
    job = client.load_table_from_dataframe(df_agg, dest_table, job_config=job_config)
    job.result()

    print(f"[ML] ✓ Done. Risk distribution:")
    print(df_agg["risk_level"].value_counts().to_string())
