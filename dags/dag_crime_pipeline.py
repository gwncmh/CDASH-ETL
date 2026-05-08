"""
dag_crime_pipeline.py
=====================
DAG chính điều phối toàn bộ Crime Analytics Pipeline.

Lịch chạy: Mỗi Chủ Nhật lúc 2:00 SA (ít traffic, GCS/BQ rẻ hơn)

Thứ tự task:
  start
    ├─► ingest_weather              (Python: NOAA API → BQ dim_weather)
    ├─► ingest_socioeconomic        (Python: Chicago Data Portal → BQ dim_socioeconomic)
    └─► create_dataproc_cluster
          ├─► etl_clean_crimes      (PySpark: làm sạch + chuẩn hóa)
          └─► etl_gis_features      (PySpark: H3 index + GIS features)
                │   [tất cả 4 tasks trên xong]
                └─► delete_dataproc_cluster
                      └─► bq_create_enriched_table   (SQL JOIN trong BQ)
                            └─► ml_train_and_predict  (Python XGBoost)
                                  └─► end
"""

import sys
import os
from datetime import datetime, timedelta
import requests as _requests

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.dataproc import (
    DataprocCreateClusterOperator,
    DataprocDeleteClusterOperator,
    DataprocSubmitJobOperator,
)
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.utils.trigger_rule import TriggerRule

# Thêm config vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../config"))
from config import *

# ─────────────────────────────────────────────
# Default args — retry 1 lần sau 5 phút nếu lỗi
# ─────────────────────────────────────────────
default_args = {
    "owner": "crime-analytics",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,  # Bật lên và điền email nếu muốn nhận alert
    "email_on_retry": False,
}

# ─────────────────────────────────────────────
# Dataproc Cluster Config
# Cluster chỉ tồn tại trong lúc job chạy để tiết kiệm chi phí
# ─────────────────────────────────────────────
CLUSTER_CONFIG = {
    "master_config": {
        "num_instances": 1,
        "machine_type_uri": DATAPROC_MASTER_TYPE,
        "disk_config": {"boot_disk_type": "pd-standard", "boot_disk_size_gb": 50},
    },
    "worker_config": {
        "num_instances": DATAPROC_NUM_WORKERS,
        "machine_type_uri": DATAPROC_WORKER_TYPE,
        "disk_config": {"boot_disk_type": "pd-standard", "boot_disk_size_gb": 50},
    },
    "software_config": {
        "image_version": "2.1-debian11",
        "properties": {
            "spark:spark.executor.memory": "4g",
            "spark:spark.driver.memory": "2g",
        },
    },
}

# ─────────────────────────────────────────────
# PySpark Job configs
# ─────────────────────────────────────────────
def make_pyspark_job(script_name: str, args: list = None) -> dict:
    """Helper tạo Dataproc PySpark job config."""
    job = {
        "reference": {"project_id": GCP_PROJECT_ID},
        "placement": {"cluster_name": DATAPROC_CLUSTER_NAME},
        "pyspark_job": {
            "main_python_file_uri": f"gs://{GCS_BUCKET}/{GCS_SCRIPTS_PATH}/{script_name}",
            "python_file_uris": [
                f"gs://{GCS_BUCKET}/{GCS_SCRIPTS_PATH}/utils.py"
            ],
            "properties": {
                "spark.jars.packages": "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.32.2"
            },
        },
    }
    if args:
        job["pyspark_job"]["args"] = args
    return job


# ─────────────────────────────────────────────
# Callable: Pull Weather từ NOAA → BQ dim_weather
# ─────────────────────────────────────────────
def _ingest_weather(ds: str, **kwargs):
    """
    Pull dữ liệu thời tiết ngày `ds` từ NOAA CDO API.
    - Dùng WRITE_APPEND để idempotent (chạy lại không duplicate nếu đã MERGE).
    - Nếu thiếu field (ngày không có tuyết, v.v.), điền 0.0 thay vì null
      để tránh JOIN bị mất dòng ở Gold layer.
    """
    import requests
    import pandas as pd
    from google.cloud import bigquery

    headers = {"token": NOAA_API_TOKEN}  # khai báo NOAA_API_TOKEN trong config.py
    params = {
        "datasetid": "GHCND",
        "stationid": NOAA_STATION_ID,    # "GHCND:USW00094846" (O'Hare, Chicago)
        "startdate": ds,
        "enddate": ds,
        "datatypeid": "TMAX,TMIN,PRCP,AWND,SNWD",
        "limit": 1000,
        "units": "standard",
    }

    resp = requests.get(
        "https://www.ncdc.noaa.gov/cdo-web/api/v2/data",
        headers=headers,
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])

    if not results:
        print(f"[ingest_weather] No data from NOAA for {ds}, skipping.")
        return

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Pivot long → wide: mỗi ngày 1 row, mỗi datatype 1 cột
    df_pivot = (
        df.pivot_table(index="date", columns="datatype", values="value", aggfunc="first")
        .reset_index()
    )

    col_map = {
        "date": "date",
        "TMAX": "temp_max",
        "TMIN": "temp_min",
        "PRCP": "precipitation",
        "AWND": "wind_speed",
        "SNWD": "snow_depth",
    }
    df_pivot = df_pivot.rename(columns=col_map)

    # Đảm bảo đủ cột, điền 0.0 nếu NOAA không trả về datatype đó
    for col in ["temp_max", "temp_min", "precipitation", "wind_speed", "snow_depth"]:
        if col not in df_pivot.columns:
            df_pivot[col] = 0.0

    df_pivot = df_pivot[
        ["date", "temp_max", "temp_min", "precipitation", "wind_speed", "snow_depth"]
    ]

    client = bigquery.Client(project=GCP_PROJECT_ID)
    table_id = f"{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE_WEATHER}"

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        schema=[
            bigquery.SchemaField("date",          "DATE"),
            bigquery.SchemaField("temp_max",      "FLOAT64"),
            bigquery.SchemaField("temp_min",      "FLOAT64"),
            bigquery.SchemaField("precipitation", "FLOAT64"),
            bigquery.SchemaField("wind_speed",    "FLOAT64"),
            bigquery.SchemaField("snow_depth",    "FLOAT64"),
        ],
    )
    client.load_table_from_dataframe(df_pivot, table_id, job_config=job_config).result()
    print(f"[ingest_weather] Loaded {len(df_pivot)} rows for {ds}.")


# ─────────────────────────────────────────────
# Callable: Pull Socioeconomic từ Chicago Data Portal → BQ
# ─────────────────────────────────────────────
def _ingest_socioeconomic(**kwargs):
    """
    Pull 77-community-area socioeconomic data từ Chicago Data Portal.
    Dùng WRITE_TRUNCATE vì đây là bảng tĩnh (cập nhật theo Census ~5 năm/lần).
    Chạy mỗi tuần nhưng ghi đè toàn bộ để đảm bảo luôn mới nhất.
    """
    import pandas as pd
    from google.cloud import bigquery

    url = "https://data.cityofchicago.org/api/views/kn9c-c2s2/rows.csv?accessType=DOWNLOAD"
    df = pd.read_csv(url)

    df = df.rename(
        columns={
            "Community Area Number":        "community_area",
            "COMMUNITY AREA NAME":          "area_name",
            "PER CAPITA INCOME ":           "per_capita_income",   # có trailing space trong nguồn gốc
            "PERCENT AGED 16+ UNEMPLOYED":  "unemployment_rate",
            "HARDSHIP INDEX":               "hardship_index",
        }
    )

    # population_density không có trong nguồn này → để null, bổ sung sau nếu cần
    df["population_density"] = None

    df = df[
        [
            "community_area",
            "area_name",
            "per_capita_income",
            "unemployment_rate",
            "hardship_index",
            "population_density",
        ]
    ].dropna(subset=["community_area"])

    df["community_area"] = df["community_area"].astype(int)

    client = bigquery.Client(project=GCP_PROJECT_ID)
    table_id = f"{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE_SOCIOECONOMIC}"

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",  # ghi đè toàn bộ vì data tĩnh
        schema=[
            bigquery.SchemaField("community_area",    "INTEGER"),
            bigquery.SchemaField("area_name",         "STRING"),
            bigquery.SchemaField("per_capita_income", "FLOAT64"),
            bigquery.SchemaField("unemployment_rate", "FLOAT64"),
            bigquery.SchemaField("hardship_index",    "INTEGER"),
            bigquery.SchemaField("population_density","FLOAT64"),
        ],
    )
    client.load_table_from_dataframe(df, table_id, job_config=job_config).result()
    print(f"[ingest_socioeconomic] Loaded {len(df)} community areas.")


# ─────────────────────────────────────────────
# SQL: Tạo enriched_crime_data bằng JOIN trong BigQuery
# ─────────────────────────────────────────────
ENRICH_SQL = f"""
CREATE OR REPLACE TABLE `{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE_ENRICHED}`
PARTITION BY DATE(date)
CLUSTER BY h3_index
AS
SELECT
  c.case_number,
  c.date,
  c.block,
  c.primary_type,
  c.latitude,
  c.longitude,
  c.h3_index,
  c.community_area,

  -- Weather features (LEFT JOIN: giữ crime record dù không có weather)
  COALESCE(w.temp_max,      0.0) AS temp_max,
  COALESCE(w.temp_min,      0.0) AS temp_min,
  COALESCE(w.precipitation, 0.0) AS precipitation,
  COALESCE(w.wind_speed,    0.0) AS wind_speed,

  -- Socioeconomic features
  s.area_name,
  s.per_capita_income,
  s.unemployment_rate,
  s.hardship_index,

  -- Derived time features cho ML
  EXTRACT(HOUR      FROM c.date) AS hour_of_day,
  EXTRACT(DAYOFWEEK FROM c.date) AS day_of_week,
  EXTRACT(MONTH     FROM c.date) AS month,

  -- Label: violent crime hay không (dùng cho classification)
  CASE
    WHEN c.primary_type IN ('ASSAULT', 'ROBBERY', 'HOMICIDE', 'BATTERY', 'KIDNAPPING')
    THEN 1 ELSE 0
  END AS is_violent

FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE_CRIMES}` c
LEFT JOIN `{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE_WEATHER}` w
  ON DATE(c.date) = w.date
LEFT JOIN `{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE_SOCIOECONOMIC}` s
  ON c.community_area = s.community_area
WHERE c.latitude  IS NOT NULL
  AND c.longitude IS NOT NULL
;
"""

# ─────────────────────────────────────────────
# ML callable — tách ra để dễ test độc lập
# ─────────────────────────────────────────────
def _run_ml_pipeline(project_id, dataset, enriched_table, predictions_table, **kwargs):
    """
    Được gọi bởi PythonOperator.
    Import lazy để worker không cần cài thư viện nếu không chạy task này.
    """
    from ml_pipeline import train_and_predict

    train_and_predict(
        project_id=project_id,
        dataset=dataset,
        enriched_table=enriched_table,
        predictions_table=predictions_table,
    )

def _export_crimes(**kwargs):
    """Gọi exporter service export enriched_crime_data → crimes.geojson"""
    r = _requests.post("http://exporter:5000/export/crimes", timeout=300)
    r.raise_for_status()
    result = r.json()
    if not result.get("ok"):
        raise RuntimeError(f"export_crimes failed: {result}")
    print(f"[export_crimes] {result['features']} features exported.")
 
 
def _export_forecast(**kwargs):
    """Gọi exporter service export prediction_results → forecast.geojson"""
    r = _requests.post("http://exporter:5000/export/forecast", timeout=120)
    r.raise_for_status()
    result = r.json()
    if not result.get("ok"):
        raise RuntimeError(f"export_forecast failed: {result}")
    print(f"[export_forecast] {result['features']} features exported.")

# ─────────────────────────────────────────────
# DAG Definition
# ─────────────────────────────────────────────
with DAG(
    dag_id="crime_analytics_pipeline",
    description="Weekly ETL + ML pipeline cho Crime Analytics Platform",
    default_args=default_args,
    schedule_interval="0 2 * * 0",  # Chủ nhật 2:00 SA
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,              # Không chạy overlap
    tags=["crime", "etl", "ml", "weekly"],
) as dag:

    # ── Bookend tasks ──────────────────────────────────────
    start = EmptyOperator(task_id="start")
    end   = EmptyOperator(task_id="end")

    # ── 1a. Ingest Weather (NOAA → BQ) ────────────────────
    # Chạy song song với create_cluster và ingest_socioeconomic
    # ds = execution date do Airflow tự truyền qua op_kwargs
    ingest_weather = PythonOperator(
        task_id="ingest_weather",
        python_callable=_ingest_weather,
        op_kwargs={"ds": "{{ ds }}"},
    )

    # ── 1b. Ingest Socioeconomic (Chicago Portal → BQ) ─────
    # Chạy song song với create_cluster và ingest_weather
    ingest_socioeconomic = PythonOperator(
        task_id="ingest_socioeconomic",
        python_callable=_ingest_socioeconomic,
    )

    # ── 2. Tạo Dataproc cluster ────────────────────────────
    # Chạy song song với 2 ingest tasks để tiết kiệm thời gian
    create_cluster = DataprocCreateClusterOperator(
        task_id="create_dataproc_cluster",
        project_id=GCP_PROJECT_ID,
        cluster_config=CLUSTER_CONFIG,
        region=GCP_REGION,
        cluster_name=DATAPROC_CLUSTER_NAME,
        gcp_conn_id=GCP_CONN_ID,
    )

    # ── 3a. PySpark ETL: làm sạch dữ liệu tội phạm thô ────
    etl_crimes = DataprocSubmitJobOperator(
        task_id="etl_clean_crimes",
        job=make_pyspark_job(
            "etl_clean_crimes.py",
            args=[
                f"--project={GCP_PROJECT_ID}",
                f"--bucket={GCS_BUCKET}",
                f"--dataset={BQ_DATASET}",
            ],
        ),
        region=GCP_REGION,
        project_id=GCP_PROJECT_ID,
        gcp_conn_id=GCP_CONN_ID,
    )

    # ── 3b. PySpark ETL: tính H3 index + GIS features ──────
    # Chạy song song với etl_crimes (độc lập nhau về input)
    etl_gis = DataprocSubmitJobOperator(
        task_id="etl_gis_features",
        job=make_pyspark_job(
            "etl_gis_features.py",
            args=[
                f"--project={GCP_PROJECT_ID}",
                f"--dataset={BQ_DATASET}",
                f"--resolution=8",
            ],
        ),
        region=GCP_REGION,
        project_id=GCP_PROJECT_ID,
        gcp_conn_id=GCP_CONN_ID,
    )

    # ── 4. Xóa cluster sau khi cả 2 ETL xong ──────────────
    # trigger_rule=ALL_DONE: xóa kể cả khi ETL lỗi, tránh tốn tiền
    delete_cluster = DataprocDeleteClusterOperator(
        task_id="delete_dataproc_cluster",
        project_id=GCP_PROJECT_ID,
        cluster_name=DATAPROC_CLUSTER_NAME,
        region=GCP_REGION,
        gcp_conn_id=GCP_CONN_ID,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # ── 5. BigQuery SQL: JOIN → enriched_crime_data ────────
    # Chờ delete_cluster + 2 ingest tasks xong mới chạy
    bq_enrich = BigQueryInsertJobOperator(
        task_id="bq_create_enriched_table",
        configuration={
            "query": {
                "query": ENRICH_SQL,
                "useLegacySql": False,
                "jobTimeoutMs": 300000,  # timeout 5 phút
            }
        },
        location=BQ_LOCATION,
        gcp_conn_id=GCP_CONN_ID,
    )

    # ── 6. ML: Train XGBoost và ghi predictions ────────────
    ml_train = PythonOperator(
        task_id="ml_train_and_predict",
        python_callable=_run_ml_pipeline,
        op_kwargs={
            "project_id":        GCP_PROJECT_ID,
            "dataset":           BQ_DATASET,
            "enriched_table":    BQ_TABLE_ENRICHED,
            "predictions_table": BQ_TABLE_PREDICTIONS,
        },
    )

    export_crimes_task = PythonOperator(
        task_id="export_crimes_geojson",
        python_callable=_export_crimes,
    )
 
    export_forecast_task = PythonOperator(
        task_id="export_forecast_geojson",
        python_callable=_export_forecast,
    )

    # ── Task Dependencies ──────────────────────────────────
    #
    #  start ──► ingest_weather ──────────────────────────────────────────┐
    #        ├─► ingest_socioeconomic ────────────────────────────────────┼──► bq_enrich ──► ml_train ──► end
    #        └─► create_cluster ──► etl_crimes ──┐                        │
    #                               etl_gis ─────┴──► delete_cluster ─────┘
    #
    start >> [ingest_weather, ingest_socioeconomic, create_cluster]

    create_cluster >> [etl_crimes, etl_gis] >> delete_cluster

    [delete_cluster, ingest_weather, ingest_socioeconomic] >> bq_enrich

    bq_enrich >> ml_train >> [export_crimes_task, export_forecast_task] >> end