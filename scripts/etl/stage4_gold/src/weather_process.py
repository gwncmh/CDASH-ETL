from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import col, to_date, broadcast, radians, lit, sin, cos, acos, row_number
from utils.logger import get_logger

logger = get_logger("Stage4_WeatherProcessor")
EARTH_RADIUS_KM = 6371.0

def process_weather_data(df_crime: DataFrame, df_weather_raw: DataFrame, df_stations: DataFrame) -> tuple[DataFrame, DataFrame]:
    logger.info("Đang xử lý dữ liệu thời tiết dạng NGANG (Wide Format)...")
    
    # ====================================================================
    # PHẦN 1: MAPPING H3 VỚI TRẠM GẦN NHẤT (Giữ nguyên logic hình học xịn của bạn)
    # ====================================================================
    df_stations_renamed = df_stations.select(
        col("ID").alias("station_id"), 
        col("LATITUDE").alias("station_lat"), 
        col("LONGITUDE").alias("station_lon")
    )

    df_unique_h3 = df_crime.filter(col("h3_index").isNotNull()) \
        .select("h3_index", "latitude", "longitude").dropDuplicates(["h3_index"])

    df_h3_cross = df_unique_h3.crossJoin(broadcast(df_stations_renamed))

    rad_h3_lat, rad_h3_lon = radians(col("latitude")), radians(col("longitude"))
    rad_st_lat, rad_st_lon = radians(col("station_lat")), radians(col("station_lon"))

    distance_col = lit(EARTH_RADIUS_KM) * acos(
        sin(rad_h3_lat) * sin(rad_st_lat) + cos(rad_h3_lat) * cos(rad_st_lat) * cos(rad_st_lon - rad_h3_lon)
    )
    df_h3_cross = df_h3_cross.withColumn("distance", distance_col)

    window_spec = Window.partitionBy("h3_index").orderBy("distance")
    df_h3_mapping = df_h3_cross.withColumn("rn", row_number().over(window_spec)) \
        .filter(col("rn") == 1).select("h3_index", "station_id")

    # ====================================================================
    # PHẦN 2: CHUẨN HÓA DỮ LIỆU THỜI TIẾT (Bỏ hoàn toàn Pivot)
    # ====================================================================
    # Khớp chính xác với các cột trong ảnh: STATION, DATE, AWND, PRCP, TMAX, TMIN
    df_weather_ready = df_weather_raw \
        .withColumnRenamed("STATION", "weather_station_id") \
        .withColumn("weather_date", to_date(col("DATE"), "M/d/yyyy")) # Format ngày dạng 1/1/2023
        
    # Chỉ lọc lấy các cột cần thiết phục vụ cho tầng Gold
    df_weather_ready = df_weather_ready.select(
        "weather_date", 
        "weather_station_id", 
        col("TMAX").alias("TMAX"), 
        col("TMIN").alias("TMIN"), 
        col("PRCP").alias("PRCP"), 
        col("AWND").alias("AWND")
    )

    return df_h3_mapping, df_weather_ready