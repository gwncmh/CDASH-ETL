"""
serving/exporter.py
-------------------
Flask API export dữ liệu từ BigQuery ra GeoJSON để Kepler.gl đọc.

Airflow gọi POST /export/all sau khi ml_train xong.
nginx serve file tĩnh tại /exports/ cho dashboard.

Endpoints:
  POST /export/crimes    → enriched_crime_data  → /exports/crimes.geojson
  POST /export/forecast  → prediction_results   → /exports/forecast.geojson
  POST /export/all       → gọi cả 2 ở trên
  GET  /status           → kiểm tra file export gần nhất
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, jsonify, request
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

PROJECT_ID = os.environ["GCP_PROJECT_ID"]
BQ_DATASET = os.environ["BQ_DATASET"]
EXPORT_DIR = Path(os.environ.get("EXPORT_DIR", "/exports"))
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

bq = bigquery.Client(project=PROJECT_ID)


# ── Helpers ────────────────────────────────────────────────────────────────

def run_query(sql: str) -> list[dict]:
    log.info("Querying BigQuery...")
    return [dict(r) for r in bq.query(sql).result()]


def to_geojson(rows: list[dict], lat="latitude", lon="longitude") -> dict:
    features = []
    for r in rows:
        lat_v = r.pop(lat, None)
        lon_v = r.pop(lon, None)
        if lat_v is None or lon_v is None:
            continue
        props = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in r.items()}
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(lon_v), float(lat_v)]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}


def write(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f)
    log.info("Wrote %s — %d features, %.1f MB", path.name,
             len(data["features"]), path.stat().st_size / 1_048_576)


# ── Queries ────────────────────────────────────────────────────────────────

def sql_crimes(days_back: int) -> str:
    since = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    return f"""
        SELECT
            case_number,
            CAST(date AS STRING)    AS date,
            primary_type,
            -- Thêm crime category để Kepler.gl dùng làm color channel
            CASE
                WHEN UPPER(primary_type) IN (
                    'BATTERY','ASSAULT','ROBBERY','HOMICIDE',
                    'CRIMINAL SEXUAL ASSAULT','SEX OFFENSE',
                    'KIDNAPPING','INTIMIDATION')
                THEN 'VIOLENT'
                WHEN UPPER(primary_type) IN (
                    'THEFT','BURGLARY','MOTOR VEHICLE THEFT',
                    'CRIMINAL DAMAGE','DECEPTIVE PRACTICE','ARSON')
                THEN 'PROPERTY'
                ELSE 'OTHER'
            END                     AS crime_category,
            community_area,
            community_area_name,
            h3_index,
            hour_of_day,
            day_of_week,
            month,
            latitude,
            longitude,
            -- Bỏ các cột ML features không cần cho visualization
            -- Giữ lại đủ để filter trên dashboard
            hardship_index,
            unemployment_rate
        FROM `{PROJECT_ID}.{BQ_DATASET}.enriched_crime_data`
        WHERE date >= '{since}'
          AND latitude  IS NOT NULL
          AND longitude IS NOT NULL
        -- Sample thay vì LIMIT cứng — đảm bảo phân bố đều theo thời gian
        -- thay vì chỉ lấy records đầu tiên
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY DATE_TRUNC(date, WEEK)
            ORDER BY RAND()
        ) <= 3000
    """


def sql_forecast() -> str:
    return f"""
        SELECT
            CAST(pr.prediction_date AS STRING) AS prediction_date,
            pr.h3_index,
            pr.community_area_name,
            pr.crime_probability,
            pr.risk_level,
            pr.dominant_type,
            pr.violent_probability,
            pr.property_probability,
            pr.confidence_level,
            pr.confidence_score,
            -- Context để tooltip hiển thị
            pr.hardship_index,
            pr.unemployment_rate,
            pr.population_density,
            pr.near_repeat_3d,
            pr.zero_streak,
            pr.roll_mean_7,
            pr.model_version,
            AVG(e.latitude)  AS latitude,
            AVG(e.longitude) AS longitude
        FROM `{PROJECT_ID}.{BQ_DATASET}.prediction_results` pr
        LEFT JOIN `{PROJECT_ID}.{BQ_DATASET}.enriched_crime_data` e
               ON pr.h3_index = e.h3_index
        WHERE pr.prediction_date = (
            SELECT MAX(prediction_date)
            FROM `{PROJECT_ID}.{BQ_DATASET}.prediction_results`
        )
        GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17
        ORDER BY pr.crime_probability DESC
    """


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/status")
def status():
    files = {}
    for name in ("crimes.geojson", "forecast.geojson"):
        p = EXPORT_DIR / name
        files[name] = {
            "exists": p.exists(),
            "size_mb": round(p.stat().st_size / 1_048_576, 2) if p.exists() else None,
            "updated_at": datetime.fromtimestamp(p.stat().st_mtime).isoformat() if p.exists() else None,
        }
    return jsonify({"status": "ok", "exports": files})


@app.post("/export/crimes")
def export_crimes():
    days_back = int(request.args.get("days", 90))
    try:
        rows = run_query(sql_crimes(days_back))
        geo  = to_geojson(rows)
        write(EXPORT_DIR / "crimes.geojson", geo)
        return jsonify({"ok": True, "features": len(geo["features"]), "days_back": days_back})
    except Exception as e:
        log.exception("export_crimes failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/export/forecast")
def export_forecast():
    try:
        rows = run_query(sql_forecast())
        geo  = to_geojson(rows)
        write(EXPORT_DIR / "forecast.geojson", geo)
        return jsonify({"ok": True, "features": len(geo["features"])})
    except Exception as e:
        log.exception("export_forecast failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/export/all")
def export_all():
    """Endpoint Airflow gọi sau khi pipeline xong."""
    results = {}

    r = export_crimes()
    results["crimes"] = r.get_json()

    r = export_forecast()
    results["forecast"] = r.get_json()

    all_ok = all(v.get("ok") for v in results.values())
    return jsonify({"ok": all_ok, **results}), (200 if all_ok else 500)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)