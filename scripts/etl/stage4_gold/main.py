import os
import sys
import yaml
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, broadcast

# --------------------------------------------------------------------
# THIẾT LẬP ĐƯỜNG DẪN HỆ THỐNG
# --------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from stage4_gold.src import (
    calculate_rolling_features,
    process_weather_data,
    clean_socio_data,
    process_poi_data
)
from utils.logger import get_logger

logger = get_logger("Stage4_Gold_Main")

def main():
    logger.info("🚀 BẮT ĐẦU PIPELINE TẦNG GOLD (DATA MART OBT) 🚀")
    
    # 1. Đọc cấu hình
    config_path = os.path.join(project_root, "config", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    env = config.get("env", "local")
    paths = config.get(env, {})
    logger.info(f"⚙️ Đang thực thi trên môi trường: [{env.upper()}]")
    
    # 2. Khởi tạo Spark
    spark = SparkSession.builder \
        .appName("ChicagoCrime_Gold_Pipeline") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
        
    try:
        # 3. Nạp dữ liệu
        logger.info("📥 Lấy dữ liệu từ tầng Silver và Raw...")
        df_crime_silver = spark.read.parquet(paths.get("silver_path"))
        df_weather_raw = spark.read.csv(paths.get("raw_weather_path"), header=True, inferSchema=True)
        df_stations = spark.read.csv(paths.get("raw_weather_station_path"), header=True, inferSchema=True)
        df_socio = spark.read.csv(paths.get("raw_socio_path"), header=True, inferSchema=True)
        df_pop = spark.read.csv(paths.get("raw_pop_path"), header=True, inferSchema=True)
        df_poi_raw = spark.read.csv(paths.get("raw_poi_path"), header=True, inferSchema=True)

        # 4. THỰC THI CÁC MODULE FEATURE ENGINEERING
        df_crime_features = calculate_rolling_features(df_crime_silver)
        df_h3_mapping, df_weather_ready = process_weather_data(df_crime_silver, df_weather_raw, df_stations)
        df_dim_socio = clean_socio_data(df_socio, df_pop)
        df_poi_mapping = process_poi_data(df_crime_silver, df_poi_raw)

        # 5. LIÊN KẾT BẢNG VÀ ĐĂNG KÝ VIEW
        logger.info("🔗 Đang tiến hành tích hợp làm phẳng dữ liệu...")
        df_fact = df_crime_features.join(broadcast(df_h3_mapping), on="h3_index", how="left")
        df_fact = df_fact.join(broadcast(df_poi_mapping), on="h3_index", how="left")
        
        # Ép kiểu thành Date (chỉ lấy phần ngày) để Join với thời tiết chuẩn xác hơn
        df_fact = df_fact.withColumn("join_date", to_date(col("incident_date_utc")))

        df_weather_ready.createOrReplaceTempView("dim_weather")
        df_dim_socio.createOrReplaceTempView("dim_socio")
        df_fact.createOrReplaceTempView("fact_crimes")

        # 6. SPARK SQL OBT (ĐÃ VÁ LỖI TÊN CỘT)
        logger.info("📝 Đang kết xuất bảng OBT cuối cùng...")
        df_gold = spark.sql("""
            SELECT 
                c.case_number,
                c.join_date as date,
                c.primary_type,
                c.community_area,
                c.h3_index,
                
                -- Đặc trưng Thời gian
                c.hour_of_day,
                c.day_of_week,
                c.month,
                
                -- Đặc trưng Thời tiết
                CAST(w.TMAX AS DOUBLE) as temp_max,
                CAST(w.PRCP AS DOUBLE) as precipitation,
                CAST(w.AWND AS DOUBLE) as wind_speed,
                
                -- Đặc trưng POI
                CAST(c.dist_nearest_school AS DOUBLE) as dist_nearest_school,
                CAST(c.dist_nearest_station AS DOUBLE) as dist_nearest_station,
                CAST(c.dist_nearest_park AS DOUBLE) as dist_nearest_park,
                CAST(c.dist_nearest_nightlife AS DOUBLE) as dist_nearest_nightlife,
                
                -- Đặc trưng Lịch sử & Bắt giữ
                CAST(c.crime_density_7d AS DOUBLE) as crime_density_7d,
                CAST(c.crime_density_30d AS DOUBLE) as crime_density_30d,
                CAST(c.arrest_rate AS DOUBLE) as arrest_rate,
                
                -- Đặc trưng Kinh tế - Xã hội
                CAST(s.unemployment_rate AS DOUBLE) as unemployment_rate,
                CAST(s.hardship_index AS INT) as hardship_index,
                CAST(s.population AS DOUBLE) as population_density
                
            FROM fact_crimes c
            LEFT JOIN dim_weather w 
                ON c.join_date = w.weather_date 
               AND c.station_id = w.weather_station_id
            -- FIX LỖI: Sửa thành s.ca_num
            LEFT JOIN dim_socio s 
                ON c.community_area = s.ca_num
        """)

        # 7. GHI FILE
        gold_output_path = paths.get("gold_path")
        logger.info(f"💾 Đang lưu dữ liệu Gold: {gold_output_path}")
        df_gold.write.mode("overwrite").parquet(gold_output_path)
        
        logger.info("🏆 PIPELINE TẦNG GOLD ĐÃ HOÀN TẤT, DỮ LIỆU ĐẠT CHUẨN 100%!")

    except Exception as e:
        logger.error(f"❌ Tiến trình thất bại tại Stage 4: {e}")
        raise e
    finally:
        spark.stop()

if __name__ == "__main__":
    main()