"""
etl_clean_crimes.py
===================
PySpark job: Đọc dữ liệu tội phạm thô từ GCS,
làm sạch và ghi vào BigQuery table fact_crimes.

Upload file này lên: gs://YOUR_BUCKET/scripts/etl_clean_crimes.py
"""

import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# ── Mapping chuẩn hóa loại tội phạm (xử lý lỗi nhập liệu) ──────────────────
CRIME_TYPE_MAP = {
    # Sai chính tả hoặc viết tắt phổ biến → chuẩn
    "THEFT FROM MOTOR VEHICLE":     "MOTOR VEHICLE THEFT",
    "CRIM SEXUAL ASSAULT":          "CRIMINAL SEXUAL ASSAULT",
    "CRIM. SEXUAL ASSAULT":         "CRIMINAL SEXUAL ASSAULT",
    "OFFENSE INVOLVING CHILDREN":   "OFFENSE INVOLVING CHILDREN",
    "NON-CRIMINAL":                 "NON - CRIMINAL",
    "NON CRIMINAL":                 "NON - CRIMINAL",
    "NARCOTICS ":                   "NARCOTICS",   # trailing space
}


def clean_crimes(spark: SparkSession, args):
    print(f"[ETL] Reading raw data from gs://{args.bucket}/{args.input_path}/")

    # ── Đọc dữ liệu thô (CSV) từ GCS ────────────────────────────────────────
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(f"gs://{args.bucket}/{args.input_path}/*.csv")
    )

    print(f"[ETL] Raw row count: {df.count()}")

    # ── 1. Chọn và đổi tên cột về chuẩn snake_case ──────────────────────────
    df = df.select(
        F.col("ID").alias("case_id"),
        F.col("Case Number").alias("case_number"),
        F.col("Date").alias("date_str"),
        F.col("Block").alias("block"),
        F.col("Primary Type").alias("primary_type_raw"),
        F.col("Description").alias("description"),
        F.col("Location Description").alias("location_description"),
        F.col("Arrest").cast("boolean").alias("arrest"),
        F.col("Domestic").cast("boolean").alias("domestic"),
        F.col("Community Area").cast("integer").alias("community_area"),
        F.col("Latitude").cast(DoubleType()).alias("latitude"),
        F.col("Longitude").cast(DoubleType()).alias("longitude"),
    )

    # ── 2. Parse datetime ────────────────────────────────────────────────────
    df = df.withColumn(
        "date",
        F.to_timestamp(F.col("date_str"), "MM/dd/yyyy hh:mm:ss a")
    ).drop("date_str")

    # ── 3. Loại bỏ dòng null ở các cột quan trọng ───────────────────────────
    before = df.count()
    df = df.dropna(subset=["case_number", "date", "latitude", "longitude"])
    after = df.count()
    print(f"[ETL] Dropped {before - after} rows with null key fields")

    # ── 4. Loại bỏ duplicate theo case_number ───────────────────────────────
    df = df.dropDuplicates(["case_number"])

    # ── 5. Filter tọa độ hợp lệ (Chicago: lat 41-43, lng -88 to -87) ────────
    df = df.filter(
        (F.col("latitude").between(41.4, 42.1)) &
        (F.col("longitude").between(-88.0, -87.2))
    )

    # ── 6. Chuẩn hóa primary_type ───────────────────────────────────────────
    # Trim whitespace + uppercase trước
    df = df.withColumn(
        "primary_type",
        F.upper(F.trim(F.col("primary_type_raw")))
    )

    # Apply mapping từ dict (dùng UDF)
    crime_map_broadcast = spark.sparkContext.broadcast(CRIME_TYPE_MAP)

    @F.udf("string")
    def normalize_crime_type(t):
        if t is None:
            return "UNKNOWN"
        mapping = crime_map_broadcast.value
        return mapping.get(t, t)  # Nếu không có trong map, giữ nguyên

    df = df.withColumn("primary_type", normalize_crime_type(F.col("primary_type")))
    df = df.drop("primary_type_raw")

    print(f"[ETL] Clean row count: {df.count()}")
    print("[ETL] Crime type distribution:")
    df.groupBy("primary_type").count().orderBy(F.desc("count")).show(20)

    # ── 7. Ghi vào BigQuery ──────────────────────────────────────────────────
    bq_table = f"{args.project}:{args.dataset}.fact_crimes"
    print(f"[ETL] Writing to BigQuery: {bq_table}")

    (
        df.write
        .format("bigquery")
        .option("table", bq_table)
        .option("temporaryGcsBucket", args.bucket)
        .option("writeMethod", "indirect")  # dùng GCS làm staging
        .mode("overwrite")                  # overwrite toàn bộ mỗi lần chạy
        .save()
    )

    print("[ETL] ✓ fact_crimes written successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project",    required=True)
    parser.add_argument("--bucket",     required=True)
    parser.add_argument("--dataset",    required=True)
    parser.add_argument("--input-path", default="data/raw/crimes", dest="input_path")
    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("CrimeETL-CleanCrimes")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    clean_crimes(spark, args)
    spark.stop()
