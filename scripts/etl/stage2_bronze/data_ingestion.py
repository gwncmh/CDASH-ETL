import argparse
import sys
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, year, month, dayofmonth, current_timestamp
from pyspark.sql.functions import coalesce
from google.cloud import storage
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, IntegerType, DoubleType

BRONZE_SCHEMA = StructType([
    StructField("unique_key",           IntegerType(), True),
    StructField("case_number",          StringType(),  True),
    StructField("date",                 StringType(),  True),
    StructField("block",                StringType(),  True),
    StructField("iucr",                 StringType(),  True),  # ← ép STRING, tránh bị infer INT
    StructField("primary_type",         StringType(),  True),
    StructField("description",          StringType(),  True),
    StructField("location_description", StringType(),  True),
    StructField("arrest",               BooleanType(), True),
    StructField("domestic",             BooleanType(), True),
    StructField("beat",                 IntegerType(), True),
    StructField("district",             IntegerType(), True),
    StructField("ward",                 IntegerType(), True),
    StructField("community_area",       IntegerType(), True),
    StructField("fbi_code",             StringType(),  True),
    StructField("x_coordinate",         DoubleType(),  True),
    StructField("y_coordinate",         DoubleType(),  True),
    StructField("year",                 IntegerType(), True),
    StructField("updated_on",           StringType(),  True),
    StructField("latitude",             DoubleType(),  True),
    StructField("longitude",            DoubleType(),  True),
    StructField("location",             StringType(),  True),
])

MANDATORY_COLUMNS = [
    "unique_key", 
    "case_number", 
    "date", 
    "primary_type", 
    "latitude", 
    "longitude",
    "community_area"
]

def raw_validator(df) -> bool:
    """Kiểm tra file thô có đúng định dạng kỳ vọng không."""
    print("Raw Validator: Đang kiểm tra tính hợp lệ của dữ liệu...")
    
    if df.isEmpty():
        raise ValueError("Raw Validator Failed: Dữ liệu đầu vào rỗng (0 dòng).")
        
    actual_columns = df.columns
    missing_cols = [c for c in MANDATORY_COLUMNS if c not in actual_columns]
    
    if missing_cols:
        raise ValueError(f"Raw Validator Failed: Thiếu các cột bắt buộc: {missing_cols}")
        
    print("Raw Validator: Passed! Dữ liệu đầy đủ cột và không rỗng.")
    return True

def partition_writer(df, bronze_path: str):
    """Tổ chức và ghi dữ liệu thô vào GCS (hoặc Local) theo cấu trúc phân vùng."""
    print("Partition Writer: Đang xử lý phân vùng dữ liệu...")
    
    # 1. Parse Date (Sử dụng đúng định dạng của dữ liệu Chicago Crime)
    df_parsed = df.withColumn(
    "parsed_date",
    coalesce(
        to_timestamp(col("date"), "yyyy-MM-dd'T'HH:mm:ss"),  # BigQuery format
        to_timestamp(col("date"), "MM/dd/yyyy hh:mm:ss a"),  # Chicago Portal format
        )
    )
    
    # 2. Bổ sung cột Partition
    df_partitioned = df_parsed \
        .withColumn("part_year", year(col("parsed_date"))) \
        .withColumn("part_month", month(col("parsed_date"))) \
        .withColumn("part_day", dayofmonth(col("parsed_date"))) \
        .withColumn("ingested_at", current_timestamp())
        
    # 3. Ghi dữ liệu Parquet
    print(f"Partition Writer: Ghi mode 'append' vào {bronze_path}")
    df_partitioned.write \
        .mode("append") \
        .partitionBy("part_year", "part_month", "part_day") \
        .parquet(bronze_path)
        
    print("Partition Writer: Ghi dữ liệu Bronze thành công!")

def file_exists_on_gcs(bucket_name, blob_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    return bucket.blob(blob_path).exists()

# ====================================================================
# ĐIỂM CHẠY CHÍNH (ENTRYPOINT)
# ====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",         type=str, default="local")
    parser.add_argument("--raw_path",    type=str, required=True)
    parser.add_argument("--bronze_path", type=str, required=True)
    # Thêm tham số ngày để chỉ xử lý 1 ngày cụ thể
    parser.add_argument("--ds",          type=str, required=False,
                        help="Ngày cần xử lý, format YYYY-MM-DD. Nếu không truyền thì full load.")
    args = parser.parse_args()

    spark = SparkSession.builder \
        .appName("Stage2_Bronze_Ingestion") \
        .config("spark.sql.session.timeZone", "UTC") \
        .getOrCreate()

    if args.ds:
        from datetime import timedelta
        d = datetime.strptime(args.ds, "%Y-%m-%d")
        
        # Đồng bộ với Stage 3: end_date = ds - 5, start_date = end_date - 21
        end_date = d - timedelta(days=5)
        start_date = end_date - timedelta(days=21)
        
        print(f"INCREMENTAL mode: xử lý khoảng {start_date.date()} → {end_date.date()}")
        
        cur = start_date
        while cur <= end_date:
            marker_path = f"raw/chicago_crime/{cur.year}/{cur.month:02d}/{cur.day:02d}/_SUCCESS"
            raw_path = f"gs://chicago-crime-raw-group15/raw/chicago_crime/{cur.year}/{cur.month:02d}/{cur.day:02d}/data.csv"
            
            if not file_exists_on_gcs("chicago-crime-raw-group15", marker_path):
                print(f"No _SUCCESS marker for {cur.date()}, skipping.")
                cur += timedelta(days=1)
                continue
            
            print(f"Processing {cur.date()}...")
            df_raw = spark.read.option("header", "true") \
                            .schema(BRONZE_SCHEMA) \
                            .csv(raw_path)
            raw_validator(df_raw)
            partition_writer(df_raw, args.bronze_path)
            cur += timedelta(days=1)

    else:
        raw_path = args.raw_path
        print("FULL LOAD mode")
        df_raw = spark.read.option("header", "true") \
                        .option("inferSchema", "true") \
                        .csv(raw_path)
        raw_validator(df_raw)
        partition_writer(df_raw, args.bronze_path)