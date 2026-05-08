# dags/tasks/pull_socioeconomic.py
import pandas as pd
from google.cloud import bigquery

# File này chứa 77 community areas của Chicago, cập nhật theo Census
SOCIOECONOMIC_URL = (
    "https://data.cityofchicago.org/api/views/kn9c-c2s2/rows.csv?accessType=DOWNLOAD"
)

def pull_socioeconomic():
    """Pull socioeconomic data từ Chicago Data Portal → BigQuery dim_socioeconomic"""
    
    df = pd.read_csv(SOCIOECONOMIC_URL)
    
    # Rename cột về đúng schema
    df = df.rename(columns={
        "Community Area Number":          "community_area",
        "COMMUNITY AREA NAME":            "community_area_name",
        "PER CAPITA INCOME ":             "per_capita_income",
        "PERCENT AGED 16+ UNEMPLOYED":    "unemployment_rate",
        "HARDSHIP INDEX":                 "hardship_index",
    })
    
    # Tính population_density từ cột dân số nếu có, hoặc để null
    df["population_density"] = None   # Census không cung cấp trực tiếp, bổ sung sau
    
    df = df[[
        "community_area", "community_area_name",
        "per_capita_income", "unemployment_rate",
        "hardship_index", "population_density"
    ]].dropna(subset=["community_area"])
    
    df["community_area"] = df["community_area"].astype(int)
    
    client = bigquery.Client()
    table_id = "your_project.cdash_warehouse.dim_socioeconomic"
    
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",  # Ghi đè toàn bộ vì data tĩnh
        schema=[
            bigquery.SchemaField("community_area",      "INTEGER"),
            bigquery.SchemaField("community_area_name", "STRING"),
            bigquery.SchemaField("per_capita_income",   "FLOAT64"),
            bigquery.SchemaField("unemployment_rate",   "FLOAT64"),
            bigquery.SchemaField("hardship_index",      "INTEGER"),
            bigquery.SchemaField("population_density",  "FLOAT64"),
        ]
    )
    
    client.load_table_from_dataframe(df, table_id, job_config=job_config).result()
    print(f"Loaded {len(df)} socioeconomic rows")