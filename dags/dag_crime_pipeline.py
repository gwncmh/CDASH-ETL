"""
dag_crime_pipeline.py
=====================
DAG chính điều phối toàn bộ Crime Analytics Pipeline.
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
    "email_on_failure": False,   # tắt cho đến khi cấu hình SMTP
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
    "initialization_actions": [
        {
            "executable_file":   f"gs://{GCS_BUCKET}/scripts/init_pip.sh",
            "execution_timeout": {"seconds": 300},
        }
    ],
}

# ── PySpark job helper ─────────────────────────────────────────────────────
def make_pyspark_job(stage_name: str, args: list = None, is_single_file: bool = False) -> dict:
    """
    Build Dataproc PySpark job config.
    """
    base = f"gs://{GCS_BUCKET}/{GCS_SCRIPTS_PATH}"
 
    if is_single_file:
        # Stage 2: file đơn, không có package zip
        main_uri         = f"{base}/{stage_name}"
        python_file_uris = [f"{base}/utils.zip"]
    else:
        # Stage 3, 4: package có src/ và utils
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

# ── Callables ──────────────────────────────────────────────────────────────

def _ingest_chicago_crime(ds: str, full_load: bool = False, **kwargs):
    """Stage 1: Extract Chicago Crime từ BigQuery Public Data → GCS."""
    sys.path.insert(0, "/opt/airflow/scripts/etl/stage1_ingestion")
    from ingest_chicago_crime import airflow_ingest_crime

    # Đọc full_load từ dag_run.conf nếu được trigger thủ công
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
    """NOAA API → BQ dim_weather. Hỗ trợ full_load qua dag_run.conf."""
    import requests
    import pandas as pd
    from google.cloud import bigquery
    from datetime import datetime, timedelta, date

    # Đọc full_load_weather từ dag_run.conf
    dag_run = kwargs.get("dag_run")
    full_load = False
    if dag_run and dag_run.conf:
        full_load = dag_run.conf.get("full_load_weather", False)

    print(f"[ingest_weather] dag_run.conf = {dag_run.conf if dag_run else 'None'}")
    print(f"[ingest_weather] full_load = {full_load}")

    # Tính date range
    if full_load:
        start_date = date(2001, 1, 1)
        end_date   = date.today() - timedelta(days=2)
        print(f"[ingest_weather] FULL LOAD mode: {start_date} → {end_date}")
    else:
        # Incremental: lấy ngày ds - 2 vì NOAA trễ 1-2 ngày
        target     = datetime.strptime(ds, "%Y-%m-%d").date() - timedelta(days=2)
        start_date = target
        end_date   = target
        print(f"[ingest_weather] INCREMENTAL mode: {start_date}")

    # NOAA giới hạn 1 năm/request → chia nhỏ theo năm
    def date_chunks(start: date, end: date):
        cur = start
        while cur <= end:
            chunk_end = min(date(cur.year, 12, 31), end)
            yield cur, chunk_end
            cur = date(cur.year + 1, 1, 1)

    all_dfs = []
    for chunk_start, chunk_end in date_chunks(start_date, end_date):
        print(f"[ingest_weather] Pulling {chunk_start} → {chunk_end} ...")
        headers = {"token": NOAA_API_TOKEN}
        params  = {
            "datasetid":  "GHCND",
            "stationid":  NOAA_STATION_ID,
            "startdate":  chunk_start.isoformat(),
            "enddate":    chunk_end.isoformat(),
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
            print(f"[ingest_weather] No data for {chunk_start} → {chunk_end}, skipping.")
            continue

        df = pd.DataFrame(results)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df_pivot = (
            df.pivot_table(index="date", columns="datatype", values="value", aggfunc="first")
            .reset_index()
            .rename(columns={
                "date": "weather_date",
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

        all_dfs.append(df_pivot[[
            "weather_date", "temp_max", "temp_min",
            "precipitation", "wind_speed", "snow_depth",
        ]])

    if not all_dfs:
        print("[ingest_weather] No data loaded. Skipping BQ write.")
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
    print(f"[ingest_weather] Loaded {len(df_final)} rows ({start_date} → {end_date}).")


def _ingest_socioeconomic(**kwargs):
    """
    US Census ACS API → aggregate lên community area → BQ dim_socioeconomic.

    Crosswalk GEOID10 → COMMAREA được đọc từ GCS (file tĩnh, upload 1 lần từ Colab).
    Không phụ thuộc Chicago Data Portal (hay bị 403).
    """
    import io
    import requests
    import pandas as pd
    from google.cloud import bigquery, storage

    CENSUS_API_KEY = "2763ffff55c7e1e1f93de60ee214acfd706926d0"

    VARS = [
        "NAME",
        "B01003_001E",   # total population
        "B19013_001E",   # median household income
        "B23025_002E",   # labor force
        "B23025_005E",   # unemployed
        "B17001_001E",   # poverty denom
        "B17001_002E",   # poverty count
    ]

    # ── 1. Đọc crosswalk từ GCS ────────────────────────────────────────────
    print("[ingest_socioeconomic] Đọc crosswalk từ GCS...")
    gcs    = storage.Client(project=GCP_PROJECT_ID)
    blob   = gcs.bucket(GCS_BUCKET).blob("raw/crosswalk/tract_to_ca.csv")
    xwalk  = pd.read_csv(io.BytesIO(blob.download_as_bytes()))

    # Chỉ giữ 2 cột cần thiết, chuẩn hóa kiểu
    xwalk["GEOID10"]  = xwalk["GEOID10"].astype(str).str.strip()
    xwalk["COMMAREA"] = pd.to_numeric(xwalk["COMMAREA"], errors="coerce")
    xwalk = xwalk[["GEOID10", "COMMAREA"]].dropna()
    xwalk["COMMAREA"] = xwalk["COMMAREA"].astype(int)
    print(f"[ingest_socioeconomic] Crosswalk: {xwalk['COMMAREA'].nunique()}/77 community areas, {len(xwalk)} tracts")

    # ── 2. Fetch ACS 2023 từ Census API ───────────────────────────────────
    print("[ingest_socioeconomic] Fetch ACS 2023...")
    url = (
        f"https://api.census.gov/data/2023/acs/acs5"
        f"?get={','.join(VARS)}"
        f"&for=tract:*"
        f"&in=state:17%20county:031"
        f"&key={CENSUS_API_KEY}"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data    = resp.json()
    raw_acs = pd.DataFrame(data[1:], columns=data[0])

    # Ép kiểu số
    for c in VARS[1:]:   # bỏ "NAME"
        raw_acs[c] = pd.to_numeric(raw_acs[c], errors="coerce")

    # Tạo GEOID10 khớp với crosswalk: "17031" + tract 6 chữ số
    raw_acs["GEOID10"] = "17031" + raw_acs["tract"].str.zfill(6)
    print(f"[ingest_socioeconomic] ACS 2023: {len(raw_acs)} tracts")

    # ── 3. Join crosswalk ─────────────────────────────────────────────────
    raw_acs = raw_acs.merge(xwalk, on="GEOID10", how="inner")
    print(f"[ingest_socioeconomic] Sau join: {len(raw_acs)} tracts / {raw_acs['COMMAREA'].nunique()} CA")

    # ── 4. Aggregate lên community area ───────────────────────────────────
    agg = raw_acs.groupby("COMMAREA").agg(
        population    = ("B01003_001E", "sum"),
        income_sum    = ("B19013_001E", "sum"),
        labor_force   = ("B23025_002E", "sum"),
        unemployed    = ("B23025_005E", "sum"),
        poverty_denom = ("B17001_001E", "sum"),
        poverty_count = ("B17001_002E", "sum"),
    ).reset_index()

    agg["unemployment_rate"] = (agg["unemployed"] / agg["labor_force"] * 100).round(2)
    agg["per_capita_income"] = (agg["income_sum"] / agg["population"]).round(0)
    agg["poverty_rate"]      = (agg["poverty_count"] / agg["poverty_denom"] * 100).round(2)
    agg["hardship_index"]    = agg["poverty_rate"].round(0).astype("Int64")

    df_final = agg.rename(columns={
        "COMMAREA":   "community_area",
        "population": "population_density",
    })[["community_area", "per_capita_income", "unemployment_rate",
        "hardship_index", "population_density"]]
    df_final["community_area_name"] = ""
    df_final = df_final[[
        "community_area", "community_area_name",
        "per_capita_income", "unemployment_rate",
        "hardship_index", "population_density",
    ]]

    # ── 5. Ghi lên BigQuery ───────────────────────────────────────────────
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
    print(f"[ingest_socioeconomic] Loaded {len(df_final)} community areas.")


def _ingest_poi(**kwargs):
    """
    OpenStreetMap (qua osmnx) → CSV → GCS.
    Stage 4 Gold đọc file này để tính dist_nearest_*.
    Chạy hàng tuần nhưng data POI thay đổi chậm nên ổn.
    """
    import pandas as pd
    from google.cloud import storage

    try:
        import osmnx as ox
    except ImportError:
        print("[ingest_poi] osmnx chưa được cài. Thêm vào _PIP_ADDITIONAL_REQUIREMENTS rồi restart.")
        raise

    poi_configs = {
        "transit_station": {"railway": "station"},
        "school":          {"amenity": "school"},
        "park":            {"leisure": "park"},
        "nightlife":       {"amenity": ["bar", "nightclub"]},
    }

    all_pois = []
    for poi_type, tags in poi_configs.items():
        print(f"[ingest_poi] Fetching {poi_type} ...")
        try:
            gdf = ox.features_from_place("Chicago, Illinois, USA", tags)
            if not gdf.empty:
                gdf["longitude"] = gdf.geometry.centroid.x
                gdf["latitude"]  = gdf.geometry.centroid.y
                gdf["poi_type"]  = poi_type
                all_pois.append(
                    pd.DataFrame(gdf[["latitude", "longitude", "poi_type"]])
                    .dropna()
                    .reset_index(drop=True)
                )
                print(f"[ingest_poi]   ✓ {len(gdf)} {poi_type}")
        except Exception as e:
            print(f"[ingest_poi]   ✗ {poi_type} failed: {e}")

    if not all_pois:
        raise RuntimeError("[ingest_poi] Không lấy được POI nào từ OpenStreetMap.")

    df = pd.concat(all_pois, ignore_index=True)
    df = df.drop_duplicates(subset=["latitude", "longitude", "poi_type"])

    # Upload lên GCS để Stage 4 Gold đọc
    gcs    = storage.Client(project=GCP_PROJECT_ID)
    bucket = gcs.bucket(GCS_BUCKET)
    blob   = bucket.blob("raw/poi/chicago_pois.csv")
    blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")
    print(f"[ingest_poi] Uploaded {len(df)} POIs → gs://{GCS_BUCKET}/raw/poi/chicago_pois.csv")


def _run_ml_pipeline(project_id, dataset, enriched_table, predictions_table, **kwargs):
    import sys, os
    sys.path.insert(0, "/opt/airflow/scripts/ml")
    from ml_pipeline import train_and_predict

    result = train_and_predict(
        project_id=project_id,
        dataset=dataset,
        enriched_table=enriched_table,
        predictions_table=predictions_table,
        lookback_days=400,          # tường minh
        write_mode="WRITE_TRUNCATE",
    )

    # Push metrics lên XCom để monitor trên Airflow UI
    ti = kwargs.get("ti")
    if ti:
        ti.xcom_push(key="val_auc",       value=result["val_auc"])
        ti.xcom_push(key="test_auc",      value=result["test_auc"])
        ti.xcom_push(key="val_f1",        value=result["val_f1"])
        ti.xcom_push(key="test_f1",       value=result["test_f1"])
        ti.xcom_push(key="model_version", value=result["model_version"])

    # Cảnh báo nếu model kém
    if result["val_auc"] < 0.75 or result["val_f1"] < 0.6:
        import logging
        logging.warning(
            "[ML] ⚠️ Model chưa đạt ngưỡng: val_auc=%.4f val_f1=%.4f",
            result["val_auc"], result["val_f1"],
        )
    return result


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
    end   = EmptyOperator(task_id="end", trigger_rule=TriggerRule.ALL_DONE)

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

    # ── Stage 1: Ingest Socioeconomic (Census ACS → BQ) ──────────────────
    ingest_socioeconomic = PythonOperator(
        task_id="ingest_socioeconomic",
        python_callable=_ingest_socioeconomic,
    )

    # ── Stage 1: Ingest POI (OpenStreetMap → GCS) ─────────────────────────
    ingest_poi = PythonOperator(
        task_id="ingest_poi",
        python_callable=_ingest_poi,
    )

    # ── Tạo Dataproc cluster (song song với ingest tasks) ─────────────────
    create_cluster = DataprocCreateClusterOperator(
        task_id="create_dataproc_cluster",
        project_id=GCP_PROJECT_ID,
        cluster_config=CLUSTER_CONFIG,
        region=GCP_REGION,
        cluster_name=DATAPROC_CLUSTER_NAME,
        gcp_conn_id=GCP_CONN_ID,
        use_if_exists=True,
    )

    # ── Stage 2: Bronze – CSV trên GCS → Parquet ─────────────────────────
    etl_bronze = DataprocSubmitJobOperator(
        task_id="etl_bronze",
        job=make_pyspark_job(
            "stage2_bronze/data_ingestion.py",
            is_single_file=True,                  # ← thêm flag này
            args=[
                "--env=prod",
                f"--raw_path=gs://{GCS_BUCKET}/raw/chicago_crime/*/*/*/data.csv",
                f"--bronze_path=gs://{GCS_BUCKET}/bronze/chicago_crime",
                "--ds={{ ds }}",
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
            "stage3_silver",
            args=[
                f"--project={GCP_PROJECT_ID}",
                f"--bronze_path=gs://{GCS_BUCKET}/bronze/chicago_crime",
                f"--silver_path=gs://{GCS_BUCKET}/silver/chicago_crime",
            ],
        ),
        region=GCP_REGION,
        project_id=GCP_PROJECT_ID,
        gcp_conn_id=GCP_CONN_ID,
    )

    # ── Stage 4: Gold – feature engineering + ghi lên BQ ─────────────────
    # ALL_DONE: chạy kể cả khi weather/socio failed (COALESCE về 0 trong SQL)
    etl_gold = DataprocSubmitJobOperator(
        task_id="etl_gold",
        job=make_pyspark_job(
            "stage4_gold",
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
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # ── Xóa cluster – chờ bronze + silver + gold đều xong ────────────────
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
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # ── Stage 6: Export GeoJSON ───────────────────────────────────────────
    export_crimes = PythonOperator(
        task_id="export_crimes_geojson",
        python_callable=_export_crimes,
    )
    export_forecast = PythonOperator(
        task_id="export_forecast_geojson",
        python_callable=_export_forecast,
    )

    # ── Dependencies ──────────────────────────────────────────────────────
    #
    #  start ──► ingest_crime ──────────────────────────────────────────────┐
    #        ├─► ingest_weather ───────────────────────────────────────────┐│
    #        ├─► ingest_socioeconomic ──────────────────────────────────── ││
    #        ├─► ingest_poi ──────────────────────────────────────────────┐ ││
    #        └─► create_cluster                                           │ ││
    #                 └─► etl_bronze ◄── ingest_crime ───────────────────┘ ││
    #                       └─► etl_silver                                  ││
    #                             └─► etl_gold ◄── ingest_weather ─────────┘│
    #                                       ◄── ingest_socioeconomic ────────┘
    #                                       ◄── ingest_poi
    #                  ┌────────────────────┴──────────────┐
    #              etl_bronze                          etl_silver
    #                  └────────────────────┬──────────────┘
    #                              delete_cluster (ALL_DONE)
    #                                       │
    #                              ml_train_and_predict
    #                               /                 \
    #                    export_crimes         export_forecast
    #                               \                 /
    #                                      end

    # Stage 1 + create cluster chạy song song
    start >> [ingest_crime, ingest_weather, ingest_socioeconomic, ingest_poi, create_cluster]

    # Bronze chờ cluster sẵn sàng VÀ crime data đã lên GCS
    [create_cluster, ingest_crime] >> etl_bronze

    # Silver chờ Bronze
    etl_bronze >> etl_silver

    # Gold chờ Silver + weather + socio + poi (ALL_DONE nên không bị block nếu 1 cái failed)
    [etl_silver, ingest_weather, ingest_socioeconomic, ingest_poi] >> etl_gold

    # delete_cluster chờ CẢ BA task Dataproc xong để tránh xóa khi đang retry
    [etl_bronze, etl_silver, etl_gold] >> delete_cluster

    # ML chỉ chạy khi ETL thành công
    delete_cluster >> ml_train

    # Export và kết thúc
    ml_train >> [export_crimes, export_forecast] >> end