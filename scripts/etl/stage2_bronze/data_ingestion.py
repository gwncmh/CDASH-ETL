import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, year, month, dayofmonth, current_timestamp

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
        to_timestamp(col("date"), "MM/dd/yyyy hh:mm:ss a")
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

# ====================================================================
# ĐIỂM CHẠY CHÍNH (ENTRYPOINT)
# ====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bronze Stage Ingestion")
    parser.add_argument("--env", type=str, default="local", help="Môi trường chạy: local hoặc prod")
    parser.add_argument("--raw_path", type=str, required=True, help="Đường dẫn file CSV gốc")
    parser.add_argument("--bronze_path", type=str, required=True, help="Đường dẫn thư mục ghi Parquet")
    
    args = parser.parse_args()

    # Khởi tạo Spark Session tự động nhận diện Local hay Cluster
    spark = SparkSession.builder \
        .appName("Stage2_Bronze_Ingestion") \
        .config("spark.sql.session.timeZone", "UTC") \
        .getOrCreate()
        
    print(f"=== BẮT ĐẦU STAGE 2: BRONZE LAYER (Môi trường: {args.env.upper()}) ===")
    
    try:
        print(f"Đang nạp dữ liệu thô từ: {args.raw_path}")
        df_raw = spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv(args.raw_path)
        
        # Chạy logic xử lý
        raw_validator(df_raw)
        partition_writer(df_raw, args.bronze_path)
        
    except Exception as e:
        print(f"Lỗi hệ thống tại Stage 2: {e}")
        raise e
    finally:
        spark.stop()
        print("=== HOÀN TẤT STAGE 2 ===")