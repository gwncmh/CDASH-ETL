"""
stage4_gold/main.py
====================
Pipeline tầng Gold: đọc Silver Parquet → feature engineering → ghi thẳng vào BigQuery.

Thay đổi so với phiên bản cũ:
- Bỏ ghi Parquet local; dùng spark-bigquery connector ghi thẳng vào
  cdash_warehouse.enriched_crime_data (PARTITION BY date, CLUSTER BY h3_index)
- Tích hợp đủ 4 nguồn feature: weather, socioeconomic, POI, rolling crime stats
- Chế độ INCREMENTAL mặc định: chỉ ghi những ngày chưa có trong BigQuery
  (dùng --mode overwrite để rebuild toàn bộ)
"""

import os
import sys
import argparse
import yaml
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, to_date, broadcast

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from stage4_gold.src import (
    calculate_rolling_features,
    process_weather_data,
    clean_socio_data,
    process_poi_data,
)
from utils.logger import get_logger

logger = get_logger("Stage4_Gold_Main")

# ── BigQuery writer ────────────────────────────────────────────────────────

def write_to_bigquery(
    df: DataFrame,
    project_id: str,
    bq_dataset: str,
    table_name: str,
    gcs_temp_bucket: str,
    write_mode: str = "append",
) -> int:
    """
    Ghi DataFrame lên BigQuery qua spark-bigquery connector.

    Args:
        write_mode: "append" (incremental) hoặc "overwrite" (rebuild toàn bộ)
    Returns:
        Số partition (ngày) đã ghi.
    """
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
        .option("writeMethod", "indirect")   # dùng GCS làm staging
        .mode(write_mode)
        .save()
    )

    # Đếm số ngày đã ghi để log
    day_count = df.select("date").distinct().count()
    logger.info("Ghi thành công %d ngày vào %s", day_count, dest)
    return day_count


def get_existing_dates(spark: SparkSession, project_id: str, bq_dataset: str,
                       table_name: str) -> set:
    """
    Đọc danh sách ngày đã có trong BigQuery để tránh ghi đè dữ liệu cũ
    (chế độ incremental).
    """
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
        logger.info("Bảng BQ chưa tồn tại hoặc rỗng (%s) → chạy Full Load.", e)
        return set()


# ── Main pipeline ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 4 Gold Pipeline")
    parser.add_argument("--mode", choices=["append", "overwrite"], default="append",
                        help="append = incremental, overwrite = rebuild toàn bộ")
    args, _ = parser.parse_known_args()

    logger.info("BẮT ĐẦU PIPELINE TẦNG GOLD (mode=%s)", args.mode.upper())

    # 1. Đọc config ──────────────────────────────────────────────────────────
    config_path = os.path.join(project_root, "config", "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    env   = config.get("env", "local")
    paths = config.get(env, {})
    bq    = config.get("bigquery", {})

    project_id      = bq.get("project_id")
    bq_dataset      = bq.get("dataset")
    bq_table        = bq.get("table_enriched", "enriched_crime_data")
    gcs_temp_bucket = bq.get("temp_bucket", paths.get("gcs_bucket"))

    logger.info("Môi trường: [%s] | Dest: %s.%s.%s", env.upper(), project_id, bq_dataset, bq_table)

    # 2. Khởi tạo Spark ──────────────────────────────────────────────────────
    spark = (
        SparkSession.builder
        .appName("ChicagoCrime_Gold_Pipeline")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        # spark-bigquery connector JAR được cung cấp qua --jars khi submit lên Dataproc
        .getOrCreate()
    )

    try:
        # 3. Nạp dữ liệu ─────────────────────────────────────────────────────
        logger.info("Đọc dữ liệu Silver từ: %s", paths["silver_path"])
        df_silver = spark.read.parquet(paths["silver_path"])

        # Lọc incremental: bỏ qua ngày đã có trong BigQuery
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

        logger.info("Đọc dữ liệu ngoại cảnh (weather, socioeconomic, POI)...")
        df_weather_raw = spark.read.csv(paths["raw_weather_path"],      header=True, inferSchema=True)
        df_stations    = spark.read.csv(paths["raw_weather_station_path"], header=True, inferSchema=True)
        df_socio       = spark.read.csv(paths["raw_socio_path"],         header=True, inferSchema=True)
        df_pop         = spark.read.csv(paths["raw_pop_path"],           header=True, inferSchema=True)
        df_poi_raw     = spark.read.csv(paths["raw_poi_path"],           header=True, inferSchema=True)

        # 4. Feature Engineering ─────────────────────────────────────────────
        logger.info("Bước 4a: Rolling crime features...")
        df_crime_feat = calculate_rolling_features(df_silver)

        logger.info("Bước 4b: Weather mapping...")
        df_h3_mapping, df_weather_ready = process_weather_data(df_silver, df_weather_raw, df_stations)

        logger.info("Bước 4c: Socioeconomic features...")
        df_dim_socio = clean_socio_data(df_socio, df_pop)

        logger.info("Bước 4d: POI features...")
        df_poi_mapping = process_poi_data(df_silver, df_poi_raw)

        # 5. JOIN tất cả lại ──────────────────────────────────────────────────
        logger.info("JOIN các bảng feature lại...")
        df_fact = (
            df_crime_feat
            .join(broadcast(df_h3_mapping),   on="h3_index",       how="left")
            .join(broadcast(df_poi_mapping),  on="h3_index",       how="left")
        )
        df_fact = df_fact.withColumn("join_date", to_date(col("incident_date_utc")))

        df_weather_ready.createOrReplaceTempView("dim_weather")
        df_dim_socio.createOrReplaceTempView("dim_socio")
        df_fact.createOrReplaceTempView("fact_crimes")

        # 6. Spark SQL – tạo bảng Gold với đầy đủ features ───────────────────
        logger.info("Tạo bảng Gold (enriched_crime_data)...")
        df_gold = spark.sql("""
            SELECT
                c.case_number,
                c.join_date                                AS date,
                c.primary_type,
                c.community_area,
                c.h3_index,
                c.latitude,
                c.longitude,

                -- Đặc trưng thời gian
                c.hour_of_day,
                c.day_of_week,
                c.month,

                -- Đặc trưng thời tiết (LEFT JOIN: giữ crime dù không có weather)
                COALESCE(CAST(w.TMAX AS DOUBLE), 0.0)     AS temp_max,
                COALESCE(CAST(w.PRCP AS DOUBLE), 0.0)     AS precipitation,
                COALESCE(CAST(w.AWND AS DOUBLE), 0.0)     AS wind_speed,

                -- Đặc trưng POI
                CAST(c.dist_nearest_school    AS DOUBLE)  AS dist_nearest_school,
                CAST(c.dist_nearest_station   AS DOUBLE)  AS dist_nearest_station,
                CAST(c.dist_nearest_park      AS DOUBLE)  AS dist_nearest_park,
                CAST(c.dist_nearest_nightlife AS DOUBLE)  AS dist_nearest_nightlife,

                -- Đặc trưng lịch sử & bắt giữ
                CAST(c.crime_density_7d       AS DOUBLE)  AS crime_density_7d,
                CAST(c.crime_density_30d      AS DOUBLE)  AS crime_density_30d,
                CAST(c.arrest_rate            AS DOUBLE)  AS arrest_rate,

                -- Đặc trưng kinh tế - xã hội
                CAST(s.unemployment_rate      AS DOUBLE)  AS unemployment_rate,
                CAST(s.hardship_index         AS INT)     AS hardship_index,
                CAST(s.population             AS DOUBLE)  AS population_density

            FROM fact_crimes c
            LEFT JOIN dim_weather w
                ON c.join_date = w.weather_date
               AND c.station_id = w.weather_station_id
            LEFT JOIN dim_socio s
                ON c.community_area = s.ca_num

            WHERE c.h3_index IS NOT NULL
              AND c.latitude  IS NOT NULL
              AND c.longitude IS NOT NULL
        """)

        # 7. Kiểm tra DQC trước khi ghi ──────────────────────────────────────
        gold_count = df_gold.count()
        if gold_count == 0:
            raise ValueError("DQC FAILED: Bảng Gold rỗng sau khi JOIN – kiểm tra dữ liệu đầu vào!")

        null_h3 = df_gold.filter(col("h3_index").isNull()).count()
        if null_h3 / gold_count > 0.05:
            raise ValueError(f"DQC FAILED: Tỷ lệ null h3_index = {null_h3/gold_count:.1%} > 5%")

        logger.info("DQC PASSED: %d bản ghi Gold hợp lệ.", gold_count)

        # 8. Ghi lên BigQuery ─────────────────────────────────────────────────
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