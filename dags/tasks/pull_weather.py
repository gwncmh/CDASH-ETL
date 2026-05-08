# dags/tasks/pull_weather.py
import requests
import pandas as pd
from google.cloud import bigquery
from datetime import datetime, timedelta

NOAA_TOKEN = "ystxZEXaKsDWcEwfgFOIZFVpIASQCHNy"
STATION_ID = "GHCND:USW00094846"

def pull_weather(start_date: str, end_date: str):
    """Pull weather data từ NOAA và load vào BigQuery dim_weather"""
    
    url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    headers = {"token": NOAA_TOKEN}
    
    params = {
        "datasetid": "GHCND",
        "stationid": STATION_ID,
        "startdate": start_date,   # format: YYYY-MM-DD
        "enddate": end_date,
        "datatypeid": "TMAX,TMIN,PRCP,AWND,SNWD",
        "limit": 1000,
        "units": "standard"        # Fahrenheit, inches
    }
    
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    results = response.json().get("results", [])
    
    if not results:
        print(f"No weather data for {start_date} to {end_date}")
        return
    
    # Pivot từ dạng long → wide (mỗi ngày 1 row)
    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df_pivot = df.pivot_table(
        index="date", columns="datatype", values="value", aggfunc="first"
    ).reset_index()
    
    # Rename cột khớp với schema dim_weather
    df_pivot = df_pivot.rename(columns={
        "date":  "weather_date",
        "TMAX":  "temp_max",
        "TMIN":  "temp_min",
        "PRCP":  "precipitation",
        "AWND":  "wind_speed",
        "SNWD":  "snow_depth"
    })
    
    # Đảm bảo đủ cột, điền 0 nếu thiếu (vd: ngày không có tuyết)
    for col in ["temp_max", "temp_min", "precipitation", "wind_speed", "snow_depth"]:
        if col not in df_pivot.columns:
            df_pivot[col] = 0.0
    
    # Load lên BigQuery
    client = bigquery.Client()
    table_id = "your_project.cdash_warehouse.dim_weather"
    
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",   # idempotent nếu dùng MERGE
        schema=[
            bigquery.SchemaField("weather_date", "DATE"),
            bigquery.SchemaField("temp_max",      "FLOAT64"),
            bigquery.SchemaField("temp_min",      "FLOAT64"),
            bigquery.SchemaField("precipitation", "FLOAT64"),
            bigquery.SchemaField("wind_speed",    "FLOAT64"),
            bigquery.SchemaField("snow_depth",    "FLOAT64"),
        ]
    )
    
    client.load_table_from_dataframe(df_pivot, table_id, job_config=job_config).result()
    print(f"Loaded {len(df_pivot)} weather rows")