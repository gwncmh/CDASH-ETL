"""
stage1_ingestion/ingest_chicago_crime.py
=========================================
Extract dữ liệu Chicago Crime từ BigQuery Public Data rồi upload lên GCS.

Chạy bởi Airflow PythonOperator trước khi Bronze stage bắt đầu.

Flow:
  bigquery-public-data.chicago_crime.crime
      → query theo date range
      → export CSV tạm lên GCS (raw/{source}/{year}/{month}/{day}/)
      → ghi file _SUCCESS marker

Usage (manual):
  python ingest_chicago_crime.py \
      --project sage-mind-489618-n5 \
      --bucket chicago-crime-raw-group15 \
      --start_date 2024-01-01 \
      --end_date   2024-01-31
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

from google.cloud import bigquery, storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("Stage1_Ingestion")

# ── Constants ──────────────────────────────────────────────────────────────
SOURCE_NAME   = "chicago_crime"
PUBLIC_TABLE  = "bigquery-public-data.chicago_crime.crime"

# Cột cần thiết – khớp với MANDATORY_COLUMNS của Bronze stage
SELECT_COLS = """
    unique_key,
    case_number,
    date,
    block,
    iucr,
    primary_type,
    description,
    location_description,
    arrest,
    domestic,
    beat,
    district,
    ward,
    community_area,
    fbi_code,
    x_coordinate,
    y_coordinate,
    year,
    updated_on,
    latitude,
    longitude,
    location
"""


# ── Helpers ────────────────────────────────────────────────────────────────

def gcs_prefix(bucket: str, year: int, month: int, day: int) -> str:
    """Trả về prefix GCS theo cấu trúc spec: raw/{source}/{year}/{month}/{day}/"""
    return f"gs://{bucket}/raw/{SOURCE_NAME}/{year:04d}/{month:02d}/{day:02d}"


def date_range(start: date, end: date):
    """Generator trả về từng ngày trong khoảng [start, end]."""
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def extract_one_day(
    bq_client: bigquery.Client,
    gcs_client: storage.Client,
    bucket_name: str,
    target_date: date,
    overwrite: bool = False,
) -> int:
    """
    Extract dữ liệu của một ngày cụ thể và upload lên GCS.

    Returns:
        Số bản ghi đã export (0 nếu không có dữ liệu hoặc đã tồn tại).
    """
    y, m, d = target_date.year, target_date.month, target_date.day
    prefix     = gcs_prefix(bucket_name, y, m, d)
    marker_key = f"raw/{SOURCE_NAME}/{y:04d}/{m:02d}/{d:02d}/_SUCCESS"

    bucket = gcs_client.bucket(bucket_name)

    # Kiểm tra _SUCCESS marker – idempotent: không chạy lại nếu đã có
    if not overwrite and bucket.blob(marker_key).exists():
        log.info("[%s] _SUCCESS marker exists, skipping.", target_date)
        return 0

    date_str = target_date.strftime("%Y-%m-%d")
    sql = f"""
        SELECT {SELECT_COLS}
        FROM `{PUBLIC_TABLE}`
        WHERE DATE(date) = '{date_str}'
        ORDER BY unique_key
    """

    log.info("[%s] Querying BigQuery...", date_str)
    query_job = bq_client.query(sql)
    rows = list(query_job.result())

    if not rows:
        log.info("[%s] No data found, skipping.", date_str)
        return 0

    # Chuyển kết quả thành CSV trong memory
    import io, csv
    buf = io.StringIO()
    writer = csv.writer(buf)

    # Header
    writer.writerow([field.name for field in rows[0].__class__._fields
                     if hasattr(rows[0].__class__, '_fields')]
                    if hasattr(rows[0].__class__, '_fields')
                    else list(rows[0].keys()))

    for row in rows:
        writer.writerow(list(row.values()))

    csv_bytes = buf.getvalue().encode("utf-8")

    # Upload CSV lên GCS
    blob_name = f"raw/{SOURCE_NAME}/{y:04d}/{m:02d}/{d:02d}/data.csv"
    blob = bucket.blob(blob_name)
    blob.upload_from_string(csv_bytes, content_type="text/csv")
    log.info("[%s] Uploaded %d rows → gs://%s/%s", date_str, len(rows), bucket_name, blob_name)

    # Ghi _SUCCESS marker
    bucket.blob(marker_key).upload_from_string(
        f"rows={len(rows)}\nsource={PUBLIC_TABLE}\ndate={date_str}\n",
        content_type="text/plain",
    )
    log.info("[%s] _SUCCESS marker written.", date_str)

    return len(rows)


def extract_range(
    project_id: str,
    bucket_name: str,
    start_date: date,
    end_date: date,
    overwrite: bool = False,
) -> dict:
    """
    Extract toàn bộ khoảng ngày.  Trả về dict tổng kết.
    Được gọi trực tiếp bởi Airflow PythonOperator.
    """
    bq_client  = bigquery.Client(project=project_id)
    gcs_client = storage.Client(project=project_id)

    total_rows  = 0
    total_days  = 0
    skipped     = 0

    for target_date in date_range(start_date, end_date):
        rows = extract_one_day(bq_client, gcs_client, bucket_name, target_date, overwrite)
        if rows > 0:
            total_rows += rows
            total_days += 1
        else:
            skipped += 1

    summary = {
        "start_date":  start_date.isoformat(),
        "end_date":    end_date.isoformat(),
        "days_loaded": total_days,
        "days_skipped": skipped,
        "total_rows":  total_rows,
    }
    log.info("Ingestion complete: %s", summary)
    return summary


# ── Airflow callable wrapper ────────────────────────────────────────────────

def airflow_ingest_crime(
    project_id: str,
    bucket_name: str,
    ds: str,           # Airflow execution date (YYYY-MM-DD)
    full_load: bool = False,
    full_load_start: str = "2001-01-01",
    **kwargs,
) -> None:
    """
    Callable cho Airflow PythonOperator.

    - full_load=True  → load toàn bộ từ 2001 đến ngày chạy (lần đầu tiên)
    - full_load=False → chỉ load ngày execution_date (chạy hàng ngày/tuần)
    """
    from datetime import datetime

    if full_load:
        start = date.fromisoformat(full_load_start)
        end   = date.today()
        log.info("FULL LOAD mode: %s → %s", start, end)
    else:
        target = date.fromisoformat(ds)
        start  = target
        end    = target
        log.info("INCREMENTAL mode: %s", target)

    extract_range(
        project_id=project_id,
        bucket_name=bucket_name,
        start_date=start,
        end_date=end,
        overwrite=False,
    )


# ── CLI entrypoint ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Ingest Chicago Crime → GCS")
    parser.add_argument("--project",    required=True, help="GCP Project ID")
    parser.add_argument("--bucket",     required=True, help="GCS bucket name (no gs://)")
    parser.add_argument("--start_date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end_date",   required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Re-download ngay cả khi đã có _SUCCESS marker")
    args = parser.parse_args()

    result = extract_range(
        project_id=args.project,
        bucket_name=args.bucket,
        start_date=date.fromisoformat(args.start_date),
        end_date=date.fromisoformat(args.end_date),
        overwrite=args.overwrite,
    )
    sys.exit(0 if result["days_loaded"] >= 0 else 1)