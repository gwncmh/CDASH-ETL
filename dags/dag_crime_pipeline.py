"""
dag_crime_pipeline.py
=====================
DAG chính điều phối toàn bộ Crime Analytics Pipeline.

Lịch chạy: Mỗi Chủ Nhật lúc 2:00 SA (UTC+7)

Thứ tự task (so với phiên bản cũ, có 2 thay đổi):
  [1] THÊM task ingest_chicago_crime (Stage 1) chạy song song với weather/socio
  [2] BỎ task bq_create_enriched_table (ENRICH_SQL) –
      Stage 4 Gold PySpark đã JOIN đủ weather + socio + POI và ghi thẳng lên BQ

Dependency graph:
  start
    ├─► ingest_chicago_crime       (Stage 1: BQ Public Data → GCS)
    ├─► ingest_weather             (NOAA → BQ dim_weather)
    ├─► ingest_socioeconomic       (Chicago Portal → BQ dim_socioeconomic)
    └─► create_dataproc_cluster
          ├─► etl_bronze           (Stage 2: CSV → Parquet Bronze)
          │     └─► etl_silver     (Stage 3: Parquet → clean Silver)
          └─► (cluster chờ silver xong)
                └─► etl_gold       (Stage 4: Silver + dim tables → BQ enriched)
                      └─► delete_dataproc_cluster
                            └─► ml_train_and_predict
                                  ├─► export_crimes_geojson
                                  └─► export_forecast_geojson
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
from airflow.utils.trigger_rule import TriggerRule

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../config"))
from config import *

# ── Default args ───────────────────────────────────────────────────────────
default_args = {
    "owner": "crime-analytics",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "email_on_retry": False,
    "email": ["cdash-alerts@example.com"],   # <── đổi thành email thật
}

# ── Dataproc Cluster Config ────────────────────────────────────────────────
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
            "spark:spark.driver.memory":   "2g",
        },
    },
}

# ── PySpark job helper ─────────────────────────────────────────────────────
def make_pyspark_job(script_name: str, args: list = None) -> dict:
    job = {
        "reference": {"project_id": GCP_PROJECT_ID},
        "placement": {"cluster_name": DATAPROC_CLUSTER_NAME},
        "pyspark_job": {
            "main_python_file_uri": f"gs://{GCS_BUCKET}/{GCS_SCRIPTS_PATH}/{script_name}",
            "python_file_uris": [
                f"gs://{GCS_BUCKET}/{GCS_SCRIPTS_PATH}/utils.py"
            ],
            "properties": {
                # spark-bigquery connector – dùng bởi Stage 4 Gold để ghi lên BQ
                "spark.jars.packages": (
                    "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.32.2"
                ),
            },
        },
    }
    if args:
        job["pyspark_job"]["args"] = args
    return job


# ── Callables ──────────────────────────────────────────────────────────────

def _ingest_chicago_crime(ds: str, full_load: bool = False, **kwargs):
    """
    Stage 1: Extract Chicago Crime từ BigQuery Public Data → GCS.
    - full_load=True  : load từ 2001 đến hôm nay (chỉ chạy lần đầu)
    - full_load=False : chỉ load ngày execution_date (chạy hàng tuần)
    """
    sys.path.insert(0, "/opt/airflow/scripts/etl/stage1_ingestion")
    from ingest_chicago_crime import airflow_ingest_crime

    airflow_ingest_crime(
        project_id=GCP_PROJECT_ID,
        bucket_name=GCS_BUCKET,
        ds=ds,
        full_load=full_load,
    )


def _ingest_weather(ds: str, **kwargs):
    """NOAA API → BQ dim_weather (giữ nguyên logic cũ)."""
    import requests
    import pandas as pd
    from google.cloud import bigquery

    headers = {"token": NOAA_API_TOKEN}
    params  = {
        "datasetid":  "GHCND",
        "stationid":  NOAA_STATION_ID,
        "startdate":  ds,
        "enddate":    ds,
        "datatypeid": "TMAX,TMIN,PRCP,AWND,SNWD",
        "limit":      1000,
        "units":      "standard",
    }
    resp = requests.get(
        "https://www.ncdc.noaa.gov/cdo-web/api/v2/data",
        headers=headers, params=params, timeout=30,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        print(f"[ingest_weather] No NOAA data for {ds}.")
        return

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df_pivot = (
        df.pivot_table(index="date", columns="datatype", values="value", aggfunc="first")
        .reset_index()
        .rename(columns={
            "date": "weather_date",   # ← tên cột khớp với schema spec
            "TMAX": "temp_max",
            "TMIN": "temp_min",
            "PRCP": "precipitation",
            "AWND": "wind_speed",
            "SNWD": "snow_depth",
        })
    )
    for col in ["temp_max", "temp_min", "precipitation", "wind_speed", "snow_depth"]:
        if col not in df_pivot.columns:
            df_pivot[col] = 0.0

    df_pivot = df_pivot[["weather_date", "temp_max", "temp_min", "precipitation", "wind_speed", "snow_depth"]]

    client   = bigquery.Client(project=GCP_PROJECT_ID)
    table_id = f"{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE_WEATHER}"
    job_cfg  = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        schema=[
            bigquery.SchemaField("weather_date",  "DATE"),
            bigquery.SchemaField("temp_max",       "FLOAT64"),
            bigquery.SchemaField("temp_min",       "FLOAT64"),
            bigquery.SchemaField("precipitation",  "FLOAT64"),
            bigquery.SchemaField("wind_speed",     "FLOAT64"),
            bigquery.SchemaField("snow_depth",     "FLOAT64"),
        ],
    )
    client.load_table_from_dataframe(df_pivot, table_id, job_config=job_cfg).result()
    print(f"[ingest_weather] Loaded {len(df_pivot)} rows for {ds}.")


def _ingest_socioeconomic(**kwargs):
    """Chicago Data Portal → BQ dim_socioeconomic (giữ nguyên logic cũ)."""
    import pandas as pd
    from google.cloud import bigquery

    url = "https://data.cityofchicago.org/api/views/kn9c-c2s2/rows.csv?accessType=DOWNLOAD"
    df  = pd.read_csv(url).rename(columns={
        "Community Area Number":       "community_area",
        "COMMUNITY AREA NAME":         "area_name",
        "PER CAPITA INCOME ":          "per_capita_income",
        "PERCENT AGED 16+ UNEMPLOYED": "unemployment_rate",
        "HARDSHIP INDEX":              "hardship_index",
    })
    df["population_density"] = None
    df = df[["community_area", "area_name", "per_capita_income",
             "unemployment_rate", "hardship_index", "population_density"]
            ].dropna(subset=["community_area"])
    df["community_area"] = df["community_area"].astype(int)

    client   = bigquery.Client(project=GCP_PROJECT_ID)
    table_id = f"{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE_SOCIOECONOMIC}"
    job_cfg  = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=[
            bigquery.SchemaField("community_area",    "INTEGER"),
            bigquery.SchemaField("area_name",         "STRING"),
            bigquery.SchemaField("per_capita_income", "FLOAT64"),
            bigquery.SchemaField("unemployment_rate", "FLOAT64"),
            bigquery.SchemaField("hardship_index",    "INTEGER"),
            bigquery.SchemaField("population_density","FLOAT64"),
        ],
    )
    client.load_table_from_dataframe(df, table_id, job_config=job_cfg).result()
    print(f"[ingest_socioeconomic] Loaded {len(df)} community areas.")


def _run_ml_pipeline(project_id, dataset, enriched_table, predictions_table, **kwargs):
    from ml_pipeline import train_and_predict
    train_and_predict(
        project_id=project_id,
        dataset=dataset,
        enriched_table=enriched_table,
        predictions_table=predictions_table,
    )


def _export_crimes(**kwargs):
    r = _requests.post("http://exporter:5000/export/crimes", timeout=300)
    r.raise_for_status()
    result = r.json()
    if not result.get("ok"):
        raise RuntimeError(f"export_crimes failed: {result}")
    print(f"[export_crimes] {result['features']} features exported.")


def _export_forecast(**kwargs):
    r = _requests.post("http://exporter:5000/export/forecast", timeout=120)
    r.raise_for_status()
    result = r.json()
    if not result.get("ok"):
        raise RuntimeError(f"export_forecast failed: {result}")
    print(f"[export_forecast] {result['features']} features exported.")


# ── DAG ────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="crime_analytics_pipeline",
    description="Weekly ETL + ML pipeline cho Crime Analytics Platform",
    default_args=default_args,
    schedule_interval="0 19 * * 6",   # Chủ nhật 2:00 SA giờ VN (19:00 UTC thứ 7)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["crime", "etl", "ml", "weekly"],
) as dag:

    start = EmptyOperator(task_id="start")
    end   = EmptyOperator(task_id="end")

    # ── Stage 1: Ingest Chicago Crime (BQ Public → GCS) ───────────────────
    ingest_crime = PythonOperator(
        task_id="ingest_chicago_crime",
        python_callable=_ingest_chicago_crime,
        op_kwargs={"ds": "{{ ds }}", "full_load": False},
    )

    # ── Stage 1: Ingest Weather (NOAA → BQ) ──────────────────────────────
    ingest_weather = PythonOperator(
        task_id="ingest_weather",
        python_callable=_ingest_weather,
        op_kwargs={"ds": "{{ ds }}"},
    )

    # ── Stage 1: Ingest Socioeconomic (Chicago Portal → BQ) ──────────────
    ingest_socioeconomic = PythonOperator(
        task_id="ingest_socioeconomic",
        python_callable=_ingest_socioeconomic,
    )

    # ── Tạo Dataproc cluster (song song với 3 ingest tasks) ───────────────
    create_cluster = DataprocCreateClusterOperator(
        task_id="create_dataproc_cluster",
        project_id=GCP_PROJECT_ID,
        cluster_config=CLUSTER_CONFIG,
        region=GCP_REGION,
        cluster_name=DATAPROC_CLUSTER_NAME,
        gcp_conn_id=GCP_CONN_ID,
    )

    # ── Stage 2: Bronze – CSV trên GCS → Parquet ─────────────────────────
    etl_bronze = DataprocSubmitJobOperator(
        task_id="etl_bronze",
        job=make_pyspark_job(
            "stage2_bronze/data_ingestion.py",
            args=[
                "--env=prod",
                f"--raw_path=gs://{GCS_BUCKET}/raw/chicago_crime",
                f"--bronze_path=gs://{GCS_BUCKET}/bronze/chicago_crime",
            ],
        ),
        region=GCP_REGION,
        project_id=GCP_PROJECT_ID,
        gcp_conn_id=GCP_CONN_ID,
    )

    # ── Stage 3: Silver – làm sạch + H3 encode ───────────────────────────
    etl_silver = DataprocSubmitJobOperator(
        task_id="etl_silver",
        job=make_pyspark_job(
            "stage3_silver/main.py",
            args=[f"--project={GCP_PROJECT_ID}"],
        ),
        region=GCP_REGION,
        project_id=GCP_PROJECT_ID,
        gcp_conn_id=GCP_CONN_ID,
    )

    # ── Stage 4: Gold – feature engineering + ghi lên BQ ─────────────────
    # Chờ Silver XONG và cả weather/socioeconomic đã có trong BQ
    etl_gold = DataprocSubmitJobOperator(
        task_id="etl_gold",
        job=make_pyspark_job(
            "stage4_gold/main.py",
            args=[
                "--mode=append",
                f"--project={GCP_PROJECT_ID}",
                f"--dataset={BQ_DATASET}",
                f"--bucket={GCS_BUCKET}",
            ],
        ),
        region=GCP_REGION,
        project_id=GCP_PROJECT_ID,
        gcp_conn_id=GCP_CONN_ID,
    )

    # ── Xóa cluster sau Gold (kể cả khi lỗi) ─────────────────────────────
    delete_cluster = DataprocDeleteClusterOperator(
        task_id="delete_dataproc_cluster",
        project_id=GCP_PROJECT_ID,
        cluster_name=DATAPROC_CLUSTER_NAME,
        region=GCP_REGION,
        gcp_conn_id=GCP_CONN_ID,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # ── Stage 5: ML – train XGBoost + predict ────────────────────────────
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

    # ── Stage 6: Export GeoJSON ───────────────────────────────────────────
    export_crimes   = PythonOperator(task_id="export_crimes_geojson",   python_callable=_export_crimes)
    export_forecast = PythonOperator(task_id="export_forecast_geojson", python_callable=_export_forecast)

    # ── Dependencies ──────────────────────────────────────────────────────
    #
    #  start ──► ingest_crime ─────────────────────────────────────────────────┐
    #        ├─► ingest_weather ──────────────────────────────────────────────┐ │
    #        ├─► ingest_socioeconomic ─────────────────────────────────────── │─┤
    #        └─► create_cluster ──► etl_bronze ──► etl_silver ──► etl_gold ──┘ │
    #                                                                  │        │
    #                                             (tất cả 4 xong)     └────────┘
    #                                                  ↓
    #                                          delete_cluster
    #                                                  ↓
    #                                           ml_train_and_predict
    #                                            /                  \
    #                               export_crimes            export_forecast
    #                                            \                  /
    #                                                    end

    start >> [ingest_crime, ingest_weather, ingest_socioeconomic, create_cluster]

    # Bronze chờ: cluster sẵn sàng VÀ crime data đã lên GCS
    [create_cluster, ingest_crime] >> etl_bronze

    etl_bronze >> etl_silver

    # Gold chờ: Silver xong VÀ weather/socio đã có trong BQ
    [etl_silver, ingest_weather, ingest_socioeconomic] >> etl_gold

    etl_gold >> delete_cluster >> ml_train >> [export_crimes, export_forecast] >> end