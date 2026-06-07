"""
dags/dag_crime_pipeline.py  (PATCHED)
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
from config import (
    GCP_PROJECT_ID, GCP_REGION, GCS_BUCKET, GCS_SCRIPTS_PATH,
    BQ_DATASET, BQ_TABLE_ENRICHED, BQ_TABLE_PREDICTIONS,
    BQ_TABLE_WEATHER, BQ_TABLE_SOCIOECONOMIC,
    DATAPROC_CLUSTER_NAME, DATAPROC_MASTER_TYPE, DATAPROC_NUM_WORKERS,
    DATAPROC_WORKER_TYPE, GCP_CONN_ID, NOAA_API_TOKEN, NOAA_STATION_ID,
    CHICAGO_COMMUNITY_AREA_NAMES
)

# ── Default args ───────────────────────────────────────────────────────────
default_args = {
    "owner": "crime-analytics",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
    "email": ["cdash-alerts@example.com"],
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
    "gce_cluster_config": {                                          # ← THÊM VÀO
        "service_account": "airflow-crime-sa@sage-mind-489618-n5.iam.gserviceaccount.com",
        "service_account_scopes": [
            "https://www.googleapis.com/auth/cloud-platform"
        ],
    },
    "initialization_actions": [
        {
            "executable_file":   f"gs://{GCS_BUCKET}/scripts/init_pip.sh",
            "execution_timeout": {"seconds": 300},
        }
    ],
}

# ── PySpark job helper ─────────────────────────────────────────────────────
def make_pyspark_job(stage_name: str, args: list = None, is_single_file: bool = False) -> dict:
    base = f"gs://{GCS_BUCKET}/{GCS_SCRIPTS_PATH}"

    if is_single_file:
        main_uri         = f"{base}/{stage_name}"
        python_file_uris = [f"{base}/utils.zip"]
    else:
        main_uri         = f"{base}/{stage_name}_main.py"
        python_file_uris = [
            f"{base}/utils.zip",
            f"{base}/{stage_name}.zip",
        ]

    job = {
        "reference": {"project_id": GCP_PROJECT_ID},
        "placement": {"cluster_name": DATAPROC_CLUSTER_NAME},
        "pyspark_job": {
            "main_python_file_uri": main_uri,
            "python_file_uris":     python_file_uris,
            "properties": {
                "spark.jars.packages": (
                    "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.32.2"
                ),
            },
        },
    }
    if args:
        job["pyspark_job"]["args"] = args
    return job


# ── GCS path shortcuts ─────────────────────────────────────────────────────
_GCS = f"gs://{GCS_BUCKET}"
_BRONZE = f"{_GCS}/bronze/chicago_crime"
_SILVER = f"{_GCS}/silver/chicago_crime"


# ── Callables (giữ nguyên từ bản gốc) ─────────────────────────────────────

def _ingest_chicago_crime(ds: str, full_load: bool = False, **kwargs):
    sys.path.insert(0, "/opt/airflow/scripts/etl/stage1_ingestion")
    from ingest_chicago_crime import airflow_ingest_crime
    dag_run = kwargs.get("dag_run")
    if dag_run and dag_run.conf:
        full_load = dag_run.conf.get("full_load", full_load)
    airflow_ingest_crime(
        project_id=GCP_PROJECT_ID,
        bucket_name=GCS_BUCKET,
        ds=ds,
        full_load=full_load,
    )


def _ingest_weather(ds: str, **kwargs):
    import requests, pandas as pd
    from google.cloud import bigquery
    from datetime import datetime, timedelta, date

    dag_run   = kwargs.get("dag_run")
    full_load = dag_run.conf.get("full_load_weather", False) if (dag_run and dag_run.conf) else False

    if full_load:
        start_date = date(2001, 1, 1)
        end_date   = date.today() - timedelta(days=2)
    else:
        target     = datetime.strptime(ds, "%Y-%m-%d").date() - timedelta(days=2)
        start_date = end_date = target

    def date_chunks(start, end):
        cur = start
        while cur <= end:
            yield cur, min(date(cur.year, 12, 31), end)
            cur = date(cur.year + 1, 1, 1)

    all_dfs = []
    for chunk_start, chunk_end in date_chunks(start_date, end_date):
        headers = {"token": NOAA_API_TOKEN}
        params  = {
            "datasetid": "GHCND", "stationid": NOAA_STATION_ID,
            "startdate": chunk_start.isoformat(), "enddate": chunk_end.isoformat(),
            "datatypeid": "TMAX,TMIN,PRCP,AWND,SNWD", "limit": 1000, "units": "standard",
        }
        resp    = requests.get("https://www.ncdc.noaa.gov/cdo-web/api/v2/data",
                               headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            continue
        df = pd.DataFrame(results)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df_pivot = (
            df.pivot_table(index="date", columns="datatype", values="value", aggfunc="first")
            .reset_index()
            .rename(columns={"date": "weather_date", "TMAX": "temp_max", "TMIN": "temp_min",
                             "PRCP": "precipitation", "AWND": "wind_speed", "SNWD": "snow_depth"})
        )
        for c in ["temp_max", "temp_min", "precipitation", "wind_speed", "snow_depth"]:
            if c not in df_pivot.columns:
                df_pivot[c] = 0.0
        all_dfs.append(df_pivot[["weather_date","temp_max","temp_min","precipitation","wind_speed","snow_depth"]])

    if not all_dfs:
        return

    df_final = pd.concat(all_dfs, ignore_index=True)
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
    client.load_table_from_dataframe(df_final, table_id, job_config=job_cfg).result()


def _ingest_socioeconomic(**kwargs):
    import io, requests, pandas as pd
    from google.cloud import bigquery, storage

    CENSUS_API_KEY = "2763ffff55c7e1e1f93de60ee214acfd706926d0"
    VARS = ["NAME","B01003_001E","B19013_001E","B23025_002E","B23025_005E","B17001_001E","B17001_002E"]

    gcs   = storage.Client(project=GCP_PROJECT_ID)
    blob  = gcs.bucket(GCS_BUCKET).blob("raw/crosswalk/tract_to_ca.csv")
    xwalk = pd.read_csv(io.BytesIO(blob.download_as_bytes()))
    xwalk["GEOID10"]  = xwalk["GEOID10"].astype(str).str.strip()
    xwalk["COMMAREA"] = pd.to_numeric(xwalk["COMMAREA"], errors="coerce")
    xwalk = xwalk[["GEOID10","COMMAREA"]].dropna()
    xwalk["COMMAREA"] = xwalk["COMMAREA"].astype(int)

    url     = (f"https://api.census.gov/data/2023/acs/acs5?get={','.join(VARS)}"
               f"&for=tract:*&in=state:17%20county:031&key={CENSUS_API_KEY}")
    resp    = requests.get(url, timeout=60); resp.raise_for_status()
    data    = resp.json()
    raw_acs = pd.DataFrame(data[1:], columns=data[0])
    for c in VARS[1:]:
        raw_acs[c] = pd.to_numeric(raw_acs[c], errors="coerce")
    raw_acs["GEOID10"] = "17031" + raw_acs["tract"].str.zfill(6)
    raw_acs = raw_acs.merge(xwalk, on="GEOID10", how="inner")

    agg = raw_acs.groupby("COMMAREA").agg(
        population    =("B01003_001E","sum"), income_sum=("B19013_001E","sum"),
        labor_force   =("B23025_002E","sum"), unemployed=("B23025_005E","sum"),
        poverty_denom =("B17001_001E","sum"), poverty_count=("B17001_002E","sum"),
    ).reset_index()
    agg["unemployment_rate"] = (agg["unemployed"]/agg["labor_force"]*100).round(2)
    agg["per_capita_income"] = (agg["income_sum"]/agg["population"]).round(0)
    agg["poverty_rate"]      = (agg["poverty_count"]/agg["poverty_denom"]*100).round(2)
    agg["hardship_index"]    = agg["poverty_rate"].round(0).astype("Int64")
    df_final = agg.rename(columns={"COMMAREA":"community_area","population":"population_density"})[
        ["community_area","per_capita_income","unemployment_rate","hardship_index","population_density"]]
    df_final["community_area_name"] = df_final["community_area"].map(CHICAGO_COMMUNITY_AREA_NAMES).fillna("")
    df_final = df_final[["community_area","community_area_name","per_capita_income",
                          "unemployment_rate","hardship_index","population_density"]]

    client   = bigquery.Client(project=GCP_PROJECT_ID)
    table_id = f"{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE_SOCIOECONOMIC}"
    job_cfg  = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=[
            bigquery.SchemaField("community_area",      "INTEGER"),
            bigquery.SchemaField("community_area_name", "STRING"),
            bigquery.SchemaField("per_capita_income",   "FLOAT64"),
            bigquery.SchemaField("unemployment_rate",   "FLOAT64"),
            bigquery.SchemaField("hardship_index",      "INTEGER"),
            bigquery.SchemaField("population_density",  "FLOAT64"),
        ],
    )
    client.load_table_from_dataframe(df_final, table_id, job_config=job_cfg).result()


def _ingest_poi(**kwargs):
    import pandas as pd
    from google.cloud import storage
    try:
        import osmnx as ox
    except ImportError:
        raise ImportError("osmnx chưa được cài. Thêm vào _PIP_ADDITIONAL_REQUIREMENTS.")

    poi_configs = {
        "transit_station": {"railway": "station"},
        "school":          {"amenity": "school"},
        "park":            {"leisure": "park"},
        "nightlife":       {"amenity": ["bar","nightclub"]},
    }
    all_pois = []
    for poi_type, tags in poi_configs.items():
        try:
            gdf = ox.features_from_place("Chicago, Illinois, USA", tags)
            if not gdf.empty:
                gdf["longitude"] = gdf.geometry.centroid.x
                gdf["latitude"]  = gdf.geometry.centroid.y
                gdf["poi_type"]  = poi_type
                all_pois.append(pd.DataFrame(gdf[["latitude","longitude","poi_type"]]).dropna().reset_index(drop=True))
        except Exception as e:
            print(f"[ingest_poi] {poi_type} failed: {e}")

    if not all_pois:
        raise RuntimeError("Không lấy được POI nào từ OpenStreetMap.")

    df = pd.concat(all_pois, ignore_index=True).drop_duplicates(subset=["latitude","longitude","poi_type"])
    gcs    = storage.Client(project=GCP_PROJECT_ID)
    bucket = gcs.bucket(GCS_BUCKET)
    bucket.blob("raw/poi/chicago_pois.csv").upload_from_string(df.to_csv(index=False), content_type="text/csv")


def _run_ml_pipeline(project_id, dataset, enriched_table, predictions_table, **kwargs):
    sys.path.insert(0, "/opt/airflow/scripts/ml")
    from ml_pipeline import train_and_predict
    result = train_and_predict(
        project_id=project_id, dataset=dataset,
        enriched_table=enriched_table, predictions_table=predictions_table,
        lookback_days=400, write_mode="WRITE_TRUNCATE",
    )
    ti = kwargs.get("ti")
    if ti:
        ti.xcom_push(key="val_auc",       value=result["val_auc"])
        ti.xcom_push(key="test_auc",      value=result["test_auc"])
        ti.xcom_push(key="val_f1",        value=result["val_f1"])
        ti.xcom_push(key="test_f1",       value=result["test_f1"])
        ti.xcom_push(key="model_version", value=result["model_version"])
    if result["val_auc"] < 0.75 or result["val_f1"] < 0.6:
        import logging
        logging.warning("[ML] ⚠️ Model chưa đạt ngưỡng: val_auc=%.4f val_f1=%.4f",
                        result["val_auc"], result["val_f1"])
    return result


def _export_crimes(**kwargs):
    r = _requests.post("http://exporter:5000/export/crimes", timeout=300)
    r.raise_for_status()
    result = r.json()
    if not result.get("ok"):
        raise RuntimeError(f"export_crimes failed: {result}")


def _export_forecast(**kwargs):
    r = _requests.post("http://exporter:5000/export/forecast", timeout=120)
    r.raise_for_status()
    result = r.json()
    if not result.get("ok"):
        raise RuntimeError(f"export_forecast failed: {result}")


# ── DAG ────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="crime_analytics_pipeline",
    description="Weekly ETL + ML pipeline cho Crime Analytics Platform",
    default_args=default_args,
    schedule_interval="0 19 * * 6",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["crime", "etl", "ml", "weekly"],
) as dag:

    start = EmptyOperator(task_id="start")
    end   = EmptyOperator(task_id="end", trigger_rule=TriggerRule.ALL_DONE)

    ingest_crime = PythonOperator(
        task_id="ingest_chicago_crime",
        python_callable=_ingest_chicago_crime,
        op_kwargs={"ds": "{{ ds }}", "full_load": False},
    )
    ingest_weather = PythonOperator(
        task_id="ingest_weather",
        python_callable=_ingest_weather,
        op_kwargs={"ds": "{{ ds }}"},
    )
    ingest_socioeconomic = PythonOperator(
        task_id="ingest_socioeconomic",
        python_callable=_ingest_socioeconomic,
    )
    ingest_poi = PythonOperator(
        task_id="ingest_poi",
        python_callable=_ingest_poi,
    )

    create_cluster = DataprocCreateClusterOperator(
        task_id="create_dataproc_cluster",
        project_id=GCP_PROJECT_ID,
        cluster_config=CLUSTER_CONFIG,
        region=GCP_REGION,
        cluster_name=DATAPROC_CLUSTER_NAME,
        gcp_conn_id=GCP_CONN_ID,
        use_if_exists=True,
    )

    # ── Stage 2: Bronze ────────────────────────────────────────────────────
    etl_bronze = DataprocSubmitJobOperator(
        task_id="etl_bronze",
        job=make_pyspark_job(
            "stage2_bronze/data_ingestion.py",
            is_single_file=True,
            args=[
                "--env=prod",
                f"--raw_path={_GCS}/raw/chicago_crime/*/*/*/data.csv",
                f"--bronze_path={_BRONZE}",
                "--ds={{ ds }}",
            ],
        ),
        region=GCP_REGION,
        project_id=GCP_PROJECT_ID,
        gcp_conn_id=GCP_CONN_ID,
    )

    # ── Stage 3: Silver  ←── PATCHED: thêm --bronze_path và --silver_path ──
    etl_silver = DataprocSubmitJobOperator(
        task_id="etl_silver",
        job=make_pyspark_job(
            "stage3_silver",
            args=[
                f"--project={GCP_PROJECT_ID}",
                f"--bronze_path={_BRONZE}",   # ← MỚI
                f"--silver_path={_SILVER}",   # ← MỚI
                "--ds={{ ds }}",
            ],
        ),
        region=GCP_REGION,
        project_id=GCP_PROJECT_ID,
        gcp_conn_id=GCP_CONN_ID,
    )

    # ── Stage 4: Gold  ←── PATCHED: --project / --dataset / --bucket đầy đủ
    etl_gold = DataprocSubmitJobOperator(
        task_id="etl_gold",
        job=make_pyspark_job(
            "stage4_gold",
            args=[
                "--mode=append",              # argparse nhận trực tiếp
                f"--project={GCP_PROJECT_ID}",
                f"--dataset={BQ_DATASET}",
                f"--bucket={GCS_BUCKET}",    # ← MỚI (không có gs://)
            ],
        ),
        region=GCP_REGION,
        project_id=GCP_PROJECT_ID,
        gcp_conn_id=GCP_CONN_ID,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    delete_cluster = DataprocDeleteClusterOperator(
        task_id="delete_dataproc_cluster",
        project_id=GCP_PROJECT_ID,
        cluster_name=DATAPROC_CLUSTER_NAME,
        region=GCP_REGION,
        gcp_conn_id=GCP_CONN_ID,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    ml_train = PythonOperator(
        task_id="ml_train_and_predict",
        python_callable=_run_ml_pipeline,
        op_kwargs={
            "project_id":        GCP_PROJECT_ID,
            "dataset":           BQ_DATASET,
            "enriched_table":    BQ_TABLE_ENRICHED,
            "predictions_table": BQ_TABLE_PREDICTIONS,
        },
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    export_crimes   = PythonOperator(task_id="export_crimes_geojson",   python_callable=_export_crimes)
    export_forecast = PythonOperator(task_id="export_forecast_geojson", python_callable=_export_forecast)

    # ── Dependencies ───────────────────────────────────────────────────────
    start >> [ingest_crime, ingest_weather, ingest_socioeconomic, ingest_poi, create_cluster]
    [create_cluster, ingest_crime] >> etl_bronze
    etl_bronze >> etl_silver
    [etl_silver, ingest_weather, ingest_socioeconomic, ingest_poi] >> etl_gold
    [etl_bronze, etl_silver, etl_gold] >> delete_cluster
    delete_cluster >> ml_train
    ml_train >> [export_crimes, export_forecast] >> end