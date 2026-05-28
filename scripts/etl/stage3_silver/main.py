"""
scripts/etl/stage3_silver/main.py  (PATCHED)
=============================================
Bỏ phụ thuộc config.yaml khi chạy trên Dataproc.
Đường dẫn được nhận qua --bronze_path / --silver_path (argparse),
khớp với args đã được DAG truyền vào qua make_pyspark_job().
"""

import os
import sys
import argparse
import shutil

from google.cloud import storage
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# ── sys.path bootstrap (giữ nguyên để import package zip) ─────────────────
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

_zip = os.path.join(os.getcwd(), "stage3_silver.zip")
if _zip not in sys.path:
    sys.path.insert(0, _zip)

from utils.logger import get_logger
from src import normalize_schema, clean_data, encode_spatial_features
from src.deduplicator import remove_existing_records

logger = get_logger("Stage3_SubmitJob")

# ===================================================================
# CÔNG TẮC DÀNH CHO LẬP TRÌNH VIÊN
# True: Xóa sạch dữ liệu Silver cũ mỗi lần chạy để test lại từ đầu
# False: Chạy thật (Chỉ nạp thêm những dòng chưa từng có)
# ===================================================================
DEV_MODE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3 Silver Pipeline")
    parser.add_argument("--project",     required=False, default="",
                        help="GCP Project ID (không bắt buộc với Silver)")
    parser.add_argument("--bronze_path", required=True,
                        help="GCS hoặc local path tới Bronze Parquet, "
                             "vd: gs://bucket/bronze/chicago_crime")
    parser.add_argument("--silver_path", required=True,
                        help="GCS hoặc local path để ghi Silver Parquet, "
                             "vd: gs://bucket/silver/chicago_crime")
    # parse_known_args để không lỗi khi Spark tự thêm --conf / --class ...
    args, _ = parser.parse_known_args()
    return args

def partition_exists(bucket_name: str, prefix: str) -> bool:
    client = storage.Client()
    blobs = list(client.list_blobs(bucket_name, prefix=prefix, max_results=1))
    return len(blobs) > 0

def main():
    args = parse_args()

    bronze_path = args.bronze_path
    silver_path = args.silver_path

    logger.info("=== BẮT ĐẦU STAGE 3: SILVER LAYER ===")
    logger.info("bronze_path : %s", bronze_path)
    logger.info("silver_path : %s", silver_path)

    # KÍCH HOẠT DEV_MODE
    if DEV_MODE:
        logger.warning("⚙️ DEV_MODE đang BẬT: Đang dọn dẹp thư mục Silver để chạy test...")
        if os.path.exists(silver_path):
            shutil.rmtree(silver_path)
            logger.info("Đã dọn sạch Silver cũ!")

    spark = (
        SparkSession.builder
        .appName("ChicagoCrime_Stage3_Silver")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--ds", required=False, default=None,
                            help="Ngày cần xử lý YYYY-MM-DD. Nếu không truyền thì full load.")
        known_args, _ = parser.parse_known_args()

        if known_args.ds:
            from datetime import datetime
            d = datetime.strptime(known_args.ds, "%Y-%m-%d")
            # Đọc đúng partition ngày đó thay vì toàn bộ Bronze
            partition_path = (
                f"{bronze_path}"
                f"/part_year={d.year}"
                f"/part_month={d.month}"
                f"/part_day={d.day}"
            )
            bucket_name = bronze_path.replace("gs://", "").split("/")[0]
            prefix = partition_path.replace(f"gs://{bucket_name}/", "")
            
            if not partition_exists(bucket_name, prefix):
                logger.info("Partition %s chưa có data. Stage 3 dừng sớm.", partition_path)
                sys.exit(0)  # Thoát bình thường, không báo lỗi
            
            logger.info(f"INCREMENTAL mode: chỉ đọc {partition_path}")
            df_bronze = spark.read.parquet(partition_path)
        else:
            logger.info(f"FULL LOAD mode: đọc toàn bộ {bronze_path}")
            df_bronze = spark.read.parquet(bronze_path)
        logger.info("Số dòng từ Bronze: %d", df_bronze.count())

        # 1. CHỐNG TRÙNG LẶP (INCREMENTAL LOAD)
        df_new    = remove_existing_records(spark, df_bronze, silver_path)
        new_count = df_new.count()
        logger.info("Số dòng MỚI TINH cần xử lý: %d", new_count)

        if new_count == 0:
            logger.info("🛑 Không có dữ liệu mới. Hệ thống dừng sớm!")
            sys.exit(0)

        # 2. PIPELINE BIẾN ĐỔI
        df_normalized = normalize_schema(df_new)
        df_cleaned    = clean_data(df_normalized)
        logger.info("Số dòng sau khi lọc rác (Clean): %d", df_cleaned.count())

        df_silver = encode_spatial_features(df_cleaned)

        # 3. KIỂM TOÁN CHẤT LƯỢNG (DQC)
        logger.info("=== BẮT ĐẦU KIỂM TOÁN DQC ===")
        total_rows = df_silver.count()
        if total_rows == 0:
            raise ValueError("DQC FAILED: Dữ liệu trống sau khi làm sạch!")

        null_h3_count = df_silver.filter(col("h3_index").isNull()).count()
        null_h3_ratio = (null_h3_count / total_rows) * 100

        logger.info("Tổng bản ghi hợp lệ : %d", total_rows)
        logger.info("Tỷ lệ lỗi H3 (Null) : %.2f%%", null_h3_ratio)

        if null_h3_ratio > 5.0:
            raise ValueError(
                f"DQC FAILED: Lỗi mã hóa H3 vượt ngưỡng ({null_h3_ratio:.2f}%)"
            )

        logger.info("✅ DQC PASSED: Dữ liệu đạt chuẩn!")

        # 4. GHI FILE (APPEND)
        logger.info("Đang ghi Parquet xuống: %s", silver_path)
        df_silver.write.mode("append").parquet(silver_path)
        logger.info("🏆 QUÁ TRÌNH GHI HOÀN TẤT THÀNH CÔNG!")

    except SystemExit:
        pass
    except Exception as e:
        logger.error("Pipeline thất bại do lỗi: %s", e)
        raise
    finally:
        spark.stop()
        logger.info("=== ĐÓNG ỨNG DỤNG ===")


if __name__ == "__main__":
    main()