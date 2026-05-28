"""
scripts/etl/stage4_gold/main.py  (FIXED v2)
==========================================
"""

import os
import sys
import argparse

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, to_date, broadcast

# ── sys.path bootstrap ─────────────────────────────────────────────────────
_zip = os.path.join(os.getcwd(), "stage4_gold.zip")
if _zip not in sys.path:
    sys.path.insert(0, _zip)

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src import (
    calculate_rolling_features,
    process_weather_data,
    clean_socio_data,
    process_poi_data,
)
from utils.logger import get_logger

logger = get_logger("Stage4_Gold_Main")


# ── Argparse ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 4 Gold Pipeline")
    parser.add_argument("--mode",       choices=["append", "overwrite"], default="append")
    parser.add_argument("--project",    required=True,  help="GCP Project ID")
    parser.add_argument("--dataset",    required=True,  help="BigQuery dataset, vd: cdash_warehouse")
    parser.add_argument("--bucket",     required=True,  help="GCS bucket (không có gs://)")
    parser.add_argument("--table_enriched",  default="enriched_crime_data")
    parser.add_argument("--silver_path",                help="Override Silver path")
    parser.add_argument("--raw_weather_path",           help="Override weather CSV path")
    parser.add_argument("--raw_weather_station_path",   help="Override station CSV path")
    parser.add_argument("--raw_socio_path",             help="Override socioeconomic CSV path")
    parser.add_argument("--raw_pop_path",               help="Override population CSV path")
    parser.add_argument("--raw_poi_path",               help="Override POI CSV path")
    args, _ = parser.parse_known_args()
    return args


def resolve_paths(args):
    b = f"gs://{args.bucket}"
    return {
        "silver_path":               args.silver_path               or f"{b}/silver/chicago_crime",
        # [FIX-1] weather.csv được export từ BQ dim_weather (đã làm ở bước trước)
        "raw_weather_path":          args.raw_weather_path          or f"{b}/raw/weather/weather.csv",
        "raw_weather_station_path":  args.raw_weather_station_path  or f"{b}/raw/weather/chicago_stations.csv",
        # [FIX-1] socioeconomic.csv được export từ BQ dim_socioeconomic
        "raw_socio_path":            args.raw_socio_path            or f"{b}/raw/socioeconomic/socioeconomic.csv",
        "raw_pop_path":              args.raw_pop_path              or f"{b}/raw/socioeconomic/raw_population.csv",
        "raw_poi_path":              args.raw_poi_path              or f"{b}/raw/socioeconomic/chicagos_poi.csv",
        "gcs_temp_bucket":           args.bucket,
    }


# ── BigQuery helpers ───────────────────────────────────────────────────────

def write_to_bigquery(
    df: DataFrame,
    project_id: str,
    bq_dataset: str,
    table_name: str,
    gcs_temp_bucket: str,
    write_mode: str = "append",
) -> int:
    dest = f"{project_id}.{bq_dataset}.{table_name}"
    logger.info("Đang ghi vào BigQuery: %s (mode=%s)", dest, write_mode)
    (
        df.write
        .format("bigquery")
        .option("table", dest)
        .option("temporaryGcsBucket", gcs_temp_bucket)
        .option("partitionField", "date")
        .option("partitionType", "DAY")
        .option("clusteredFields", "h3_index")
        .option("writeMethod", "indirect")
        .mode(write_mode)
        .save()
    )
    day_count = df.select("date").distinct().count()
    logger.info("Ghi thành công %d ngày vào %s", day_count, dest)
    return day_count


def get_existing_dates(
    spark: SparkSession,
    project_id: str,
    bq_dataset: str,
    table_name: str,
) -> set:
    dest = f"{project_id}.{bq_dataset}.{table_name}"
    try:
        df_existing = (
            spark.read
            .format("bigquery")
            .option("table", dest)
            .load()
            .select("date")
            .distinct()
        )
        existing = {str(r["date"]) for r in df_existing.collect()}
        logger.info("BigQuery đã có %d ngày dữ liệu.", len(existing))
        return existing
    except Exception as e:
        logger.info("Bảng BQ chưa tồn tại hoặc rỗng (%s) → Full Load.", e)
        return set()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args  = parse_args()
    paths = resolve_paths(args)

    project_id      = args.project
    bq_dataset      = args.dataset
    bq_table        = args.table_enriched
    gcs_temp_bucket = paths["gcs_temp_bucket"]

    logger.info("BẮT ĐẦU PIPELINE TẦNG GOLD (mode=%s)", args.mode.upper())
    logger.info("project=%s  dataset=%s  table=%s", project_id, bq_dataset, bq_table)
    for k, v in paths.items():
        logger.info("  %-30s = %s", k, v)

    spark = (
        SparkSession.builder
        .appName("ChicagoCrime_Gold_Pipeline")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    try:
        # ── Đọc Silver ────────────────────────────────────────────────────
        logger.info("Đọc Silver từ: %s", paths["silver_path"])
        df_silver = spark.read.parquet(paths["silver_path"])

        # Log schema để dễ debug
        logger.info("Silver schema: %s", df_silver.columns)

        # Incremental: bỏ qua ngày đã có trong BigQuery
        if args.mode == "append":
            existing_dates = get_existing_dates(spark, project_id, bq_dataset, bq_table)
            if existing_dates:
                df_silver = df_silver.filter(
                    ~to_date(col("incident_date_utc")).cast("string").isin(existing_dates)
                )
                new_count = df_silver.count()
                logger.info("Sau lọc incremental: còn %d bản ghi cần xử lý.", new_count)
                if new_count == 0:
                    logger.info("Không có dữ liệu mới. Pipeline dừng sớm.")
                    return

        # ── Đọc dữ liệu ngoại cảnh ───────────────────────────────────────
        logger.info("Đọc dữ liệu ngoại cảnh...")
        df_weather_raw = spark.read.csv(paths["raw_weather_path"],         header=True, inferSchema=True)
        df_stations    = spark.read.csv(paths["raw_weather_station_path"], header=True, inferSchema=True)
        df_socio       = spark.read.csv(paths["raw_socio_path"],           header=True, inferSchema=True)
        df_poi_raw     = spark.read.csv(paths["raw_poi_path"],             header=True, inferSchema=True)

        # Log schema để kiểm tra cột có đúng không
        logger.info("Weather CSV columns: %s", df_weather_raw.columns)
        logger.info("Socio CSV columns:   %s", df_socio.columns)

        # ── Feature Engineering ───────────────────────────────────────────
        logger.info("Bước 4a: Rolling crime features...")
        df_crime_feat = calculate_rolling_features(df_silver)

        # [FIX-1] process_weather_data trả về (df_h3_mapping, df_weather_ready)
        # df_h3_mapping: h3_index → station_id (dùng để resolve station cho mỗi ô H3)
        # df_weather_ready: station_id + weather_date + TMAX/TMIN/PRCP/AWND
        logger.info("Bước 4b: Weather mapping...")
        df_h3_mapping, df_weather_ready = process_weather_data(df_silver, df_weather_raw, df_stations)

        # [FIX-1] JOIN station_id vào crime TRƯỚC khi createOrReplaceTempView
        # Không JOIN trực tiếp weather ở đây — để Spark SQL làm qua TempView
        # vì cần kết hợp cả join_date + station_id
        df_crime_feat = df_crime_feat.join(
            broadcast(df_h3_mapping), on="h3_index", how="left"
        )

        # [FIX-3] Tính POI features và JOIN vào crime_feat
        logger.info("Bước 4c: POI features...")
        df_poi_mapping = process_poi_data(df_silver, df_poi_raw)
        df_crime_feat = df_crime_feat.join(
            broadcast(df_poi_mapping), on="h3_index", how="left"
        )

        # [FIX-2] Socioeconomic — dim_socioeconomic từ BQ export đã có đủ cột
        # bao gồm per_capita_income mà ML pipeline cần
        logger.info("Bước 4d: Socioeconomic features...")
        # Khi export từ BQ dim_socioeconomic, schema đã đúng nên không cần
        # clean_socio_data (vốn join thêm population từ file riêng).
        # Dùng trực tiếp nếu file export đủ cột, hoặc dùng hàm nếu có raw_pop_path.
        try:
            df_pop = spark.read.csv(paths["raw_pop_path"], header=True, inferSchema=True)
            df_dim_socio = clean_socio_data(df_socio, df_pop)
        except Exception as e:
            logger.warning("Không đọc được raw_pop_path (%s), dùng socio CSV trực tiếp.", e)
            # Fallback: dùng thẳng file export từ BQ (đã có đủ cột)
            df_dim_socio = df_socio

        # ── Tạo TempView ─────────────────────────────────────────────────
        df_crime_feat = df_crime_feat.withColumn("join_date", to_date(col("incident_date_utc")))
        df_weather_ready.createOrReplaceTempView("dim_weather")
        df_dim_socio.createOrReplaceTempView("dim_socio")
        df_crime_feat.createOrReplaceTempView("fact_crimes")

        # Log để kiểm tra station_id đã có chưa
        logger.info("fact_crimes columns: %s", df_crime_feat.columns)

        # ── Spark SQL ─────────────────────────────────────────────────────
        logger.info("Tạo bảng Gold (enriched_crime_data)...")
        df_gold = spark.sql("""
            SELECT
                c.case_number,
                c.join_date                                          AS date,
                c.primary_type,
                c.community_area,
                c.h3_index,
                c.latitude,
                c.longitude,

                -- Đặc trưng thời gian
                c.hour_of_day,
                c.day_of_week,
                c.month,

                -- [FIX-4] Thời tiết: dùng station_id đã JOIN vào fact_crimes
                -- AWND = wind_speed trong schema NOAA
                COALESCE(CAST(w.TMAX AS DOUBLE), 0.0)               AS temp_max,
                COALESCE(CAST(w.PRCP AS DOUBLE), 0.0)               AS precipitation,
                COALESCE(CAST(w.AWND AS DOUBLE), 0.0)               AS wind_speed,

                -- [FIX-3] POI: đọc từ df_poi_mapping đã JOIN vào fact_crimes
                -- [FIX-5] COALESCE tránh NULL làm hỏng ML
                COALESCE(CAST(c.dist_nearest_school    AS DOUBLE), 1.5)  AS dist_nearest_school,
                COALESCE(CAST(c.dist_nearest_station   AS DOUBLE), 2.0)  AS dist_nearest_station,
                COALESCE(CAST(c.dist_nearest_park      AS DOUBLE), 1.2)  AS dist_nearest_park,
                COALESCE(CAST(c.dist_nearest_nightlife AS DOUBLE), 2.5)  AS dist_nearest_nightlife,

                -- Rolling crime stats
                COALESCE(CAST(c.crime_density_7d       AS DOUBLE), 0.0)  AS crime_density_7d,
                COALESCE(CAST(c.crime_density_30d      AS DOUBLE), 0.0)  AS crime_density_30d,
                COALESCE(CAST(c.arrest_rate            AS DOUBLE), 0.0)  AS arrest_rate,

                -- [FIX-2] Kinh tế - xã hội: thêm per_capita_income mà ML pipeline cần
                -- [FIX-5] COALESCE tránh NULL
                COALESCE(CAST(s.unemployment_rate      AS DOUBLE), 0.0)  AS unemployment_rate,
                COALESCE(CAST(s.hardship_index         AS INT),    0)    AS hardship_index,
                COALESCE(CAST(s.population_density     AS DOUBLE), 0.0)  AS population_density,
                COALESCE(CAST(s.per_capita_income      AS DOUBLE), 0.0)  AS per_capita_income

            FROM fact_crimes c
            -- [FIX-1] JOIN weather dùng station_id đã resolve từ df_h3_mapping
            LEFT JOIN dim_weather w
                ON c.join_date   = w.weather_date
               AND c.station_id  = w.weather_station_id
            -- JOIN socioeconomic theo community_area
            LEFT JOIN dim_socio s
                ON CAST(c.community_area AS INT) = CAST(s.community_area AS INT)

            WHERE c.h3_index IS NOT NULL
              AND c.latitude  IS NOT NULL
              AND c.longitude IS NOT NULL
        """)

        # ── DQC ───────────────────────────────────────────────────────────
        gold_count = df_gold.count()
        if gold_count == 0:
            raise ValueError("DQC FAILED: Bảng Gold rỗng sau khi JOIN — kiểm tra dữ liệu đầu vào!")

        null_h3 = df_gold.filter(col("h3_index").isNull()).count()
        if null_h3 / gold_count > 0.05:
            raise ValueError(
                f"DQC FAILED: Tỷ lệ null h3_index = {null_h3/gold_count:.1%} > 5%"
            )

        # Kiểm tra thêm weather JOIN có hoạt động không
        null_weather = df_gold.filter(col("temp_max") == 0.0).count()
        weather_null_pct = null_weather / gold_count * 100
        logger.info(
            "DQC PASSED: %d bản ghi Gold hợp lệ. "
            "Weather null/zero: %.1f%% (nếu > 50%% thì kiểm tra lại station CSV).",
            gold_count, weather_null_pct
        )

        # ── Ghi lên BigQuery ──────────────────────────────────────────────
        write_to_bigquery(
            df=df_gold,
            project_id=project_id,
            bq_dataset=bq_dataset,
            table_name=bq_table,
            gcs_temp_bucket=gcs_temp_bucket,
            write_mode=args.mode,
        )

        logger.info("PIPELINE TẦNG GOLD HOÀN TẤT!")

    except Exception as e:
        logger.error("Pipeline thất bại: %s", e)
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()