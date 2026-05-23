import requests
import pandas as pd

def fetch_socioeconomic() -> pd.DataFrame:
    url = "https://data.cityofchicago.org/resource/kn9c-c2s2.json?$limit=100"
    headers = {"X-App-Token": "r9vE8urT836pTMTzEA1gn8xd0"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    return df

def transform_socioeconomic(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.rename(columns={
        "community_area_number":        "community_area",
        "community_area_name":          "community_area_name",
        "per_capita_income_":           "per_capita_income",
        "percent_aged_16_unemployed":   "unemployment_rate",
        "hardship_index":               "hardship_index",
    })
    df["community_area"]    = pd.to_numeric(df["community_area"], errors="coerce")
    df["per_capita_income"] = pd.to_numeric(df["per_capita_income"], errors="coerce")
    df["unemployment_rate"] = pd.to_numeric(df["unemployment_rate"], errors="coerce")
    df["hardship_index"]    = pd.to_numeric(df["hardship_index"], errors="coerce")
    df["population_density"] = None

    keep_cols = [
        "community_area", "community_area_name",
        "per_capita_income", "unemployment_rate",
        "hardship_index", "population_density"
    ]
    return df[keep_cols].sort_values("community_area").reset_index(drop=True)

# --- Test ---
raw_socio = fetch_socioeconomic()
print(f"Status OK — {len(raw_socio)} rows")
print(f"Raw columns: {raw_socio.columns.tolist()}")

socio_df = transform_socioeconomic(raw_socio)
print(f"\nSau transform: {len(socio_df)} khu vực")  # Kỳ vọng: 77
print(socio_df.head())