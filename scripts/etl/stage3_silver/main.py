import os
import sys
import yaml
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

print("=== sys.path ===")
for p in sys.path:
    print(p)
print("================")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Add zip vào sys.path ngay lập tức
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

def main():
    logger.info("=== BẮT ĐẦU STAGE 3: SILVER LAYER ===")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bronze_path", required=True)
    parser.add_argument("--silver_path", required=True)
    parser.add_argument("--project",     required=False, default=GCP_PROJECT_ID)
    args, _ = parser.parse_known_args()
    bronze_path = args.bronze_path
    silver_path = args.silver_path
    
    # KÍCH HOẠT DEV_MODE
    if DEV_MODE:
        logger.warning("⚙️ DEV_MODE đang BẬT: Đang dọn dẹp thư mục Silver để chạy test...")
        if os.path.exists(silver_path):
            shutil.rmtree(silver_path)
            logger.info("Đã dọn sạch Silver cũ!")
    
    spark = SparkSession.builder \
        .appName("ChicagoCrime_Stage3_Silver") \
        .master("local[*]") \
        .config("spark.sql.session.timeZone", "UTC") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
        
    try:
        logger.info(f"Đang đọc dữ liệu từ Bronze: {bronze_path}")
        df_bronze = spark.read.parquet(bronze_path)
        logger.info(f"Số dòng từ Bronze: {df_bronze.count()}")
        
        # 1. CHỐNG TRÙNG LẶP (INCREMENTAL LOAD)
        df_new = remove_existing_records(spark, df_bronze, silver_path)
        new_count = df_new.count()
        logger.info(f"Số dòng MỚI TINH cần xử lý: {new_count}")
        
        # KIỂM TRA THOÁT SỚM BẰNG SYS.EXIT CỰC KỲ AN TOÀN
        if new_count == 0:
            logger.info("🛑 Không có dữ liệu mới. HỆ THỐNG DỪNG SỚM một cách êm ái!")
            sys.exit(0) 
            
        # 2. PIPELINE BIẾN ĐỔI
        df_normalized = normalize_schema(df_new)
        df_cleaned = clean_data(df_normalized)
        logger.info(f"Số dòng sau khi lọc rác (Clean): {df_cleaned.count()}")
        
        df_silver = encode_spatial_features(df_cleaned)
        
        # 3. KIỂM TOÁN CHẤT LƯỢNG (DQC)
        logger.info("=== BẮT ĐẦU KIỂM TOÁN DQC ===")
        total_rows = df_silver.count()
        if total_rows == 0:
            raise ValueError("DQC FAILED: Dữ liệu trống sau khi làm sạch!")
            
        null_h3_count = df_silver.filter(col("h3_index").isNull()).count()
        null_h3_ratio = (null_h3_count / total_rows) * 100
        
        logger.info(f"Tổng bản ghi hợp lệ: {total_rows}")
        logger.info(f"Tỷ lệ lỗi mã hóa H3 (Null): {null_h3_ratio:.2f}%")
        
        if null_h3_ratio > 5.0:
            raise ValueError(f"DQC FAILED: Lỗi mã hóa H3 vượt ngưỡng ({null_h3_ratio:.2f}%)")
            
        logger.info("✅ DQC PASSED: Dữ liệu đạt chuẩn!")
        
        # 4. GHI FILE (APPEND)
        logger.info(f"Đang ghi Parquet xuống: {silver_path}")
        df_silver.write.mode("append").parquet(silver_path)
        logger.info("🏆 QUÁ TRÌNH GHI HOÀN TẤT THÀNH CÔNG!")
        
    except SystemExit:
        # Bắt lệnh sys.exit(0) để không báo lỗi màu đỏ
        pass
    except Exception as e:
        logger.error(f"Pipeline thất bại do lỗi: {e}")
        raise e
    finally:
        spark.stop()
        logger.info("=== ĐÓNG ỨNG DỤNG ===")

if __name__ == "__main__":
    main()