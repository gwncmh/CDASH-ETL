import os
import sys
import yaml
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# --------------------------------------------------------------------
# THIẾT LẬP ĐƯỜNG DẪN HỆ THỐNG
# --------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.logger import get_logger
logger = get_logger("Stage4_ReadGold")

def main():
    # 1. Đọc cấu hình đường dẫn
    config_path = os.path.join(project_root, "config", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    env = config.get("env", "local")
    paths = config.get(env, {})
    gold_path = paths.get("gold_path")
    
    logger.info(f"🔎 Đang nạp dữ liệu Gold từ môi trường: [{env.upper()}]")
    logger.info(f"📂 Thư mục: {gold_path}")
    
    # 2. Khởi tạo Spark Session
    spark = SparkSession.builder \
        .appName("ChicagoCrime_ReadGold_Final") \
        .master("local[*]") \
        .getOrCreate()
        
    try:
        # 3. ĐỌC DỮ LIỆU TẦNG GOLD
        df_gold = spark.read.parquet(gold_path)
        print("\n" + "="*90)
        
        # ---------------------------------------------------------
        # BÀI KIỂM TRA SỐ 1: KIỂM TOÁN LẠI SCHEMA ĐẶC TẢ
        # ---------------------------------------------------------
        logger.info("📜 1. KIỂM TRA SCHEMA (Đảm bảo đã ép chuẩn DOUBLE/INT cho BigQuery):")
        df_gold.printSchema()
        print("="*90)
        
        # ---------------------------------------------------------
        # BÀI KIỂM TRA SỐ 2: ĐẾM TỔNG SỐ BẢN GHI
        # ---------------------------------------------------------
        total_rows = df_gold.count()
        logger.info(f"📊 2. TỔNG SỐ BẢN GHI TRONG DATA MART: {total_rows:,} dòng")
        print("="*90)
        
        # ---------------------------------------------------------
        # BÀI KIỂM TRA SỐ 3: TỔNG DUYỆT CÁC ĐẶC TRƯNG SIÊU VIỆT (WEATHER & POI)
        # ---------------------------------------------------------
        logger.info("🎯 3. HIỂN THỊ DỮ LIỆU MẪU (Tập trung soi POI và Thời tiết):")
        # Chọn ra những cột tinh túy nhất để ngắm nhìn thành quả
        df_gold.select(
            "case_number", 
            "date",
            "temp_max",
            "crime_density_7d",
            "dist_nearest_school", 
            "dist_nearest_station",
            "dist_nearest_park",
            "dist_nearest_nightlife"
        ).show(20, truncate=False)
        print("="*90)
        
        # ---------------------------------------------------------
        # BÀI KIỂM TRA SỐ 4: BÁO CÁO TỶ LỆ KHỚP NỐI THỜI TIẾT
        # ---------------------------------------------------------
        logger.info("🎯 4. THỐNG KÊ MỨC ĐỘ BAO PHỦ CỦA DỮ LIỆU THỜI TIẾT:")
        df_weather_matched = df_gold.filter(col("temp_max").isNotNull())
        match_count = df_weather_matched.count()
        
        logger.info(f"Tìm thấy {match_count:,} / {total_rows:,} vụ án khớp thành công với Thời tiết.")
        if match_count > 0:
            logger.info("-> Trích xuất 5 dòng có thời tiết để chứng minh:")
            df_weather_matched.select("date", "temp_max", "precipitation", "wind_speed").show(5)
        print("="*90 + "\n")
        
    except Exception as e:
        logger.error(f"❌ Thất bại khi đọc dữ liệu Gold: {e}")
    finally:
        spark.stop()
        logger.info("=== ĐÓNG ỨNG DỤNG ===")

if __name__ == "__main__":
    main()