from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import col, count, hour, dayofweek, month, avg, when, coalesce, lit

def calculate_rolling_features(df_crime: DataFrame) -> DataFrame:
    print("Crime Aggregator: Đang tính toán đặc trưng Lịch sử và Thời gian...")
    
    # 1. Bóc tách Đặc trưng Thời gian (Temporal Features)
    df = df_crime.withColumn("hour_of_day", hour(col("incident_date_utc"))) \
                 .withColumn("month", month(col("incident_date_utc"))) \
                 .withColumn("day_of_week", ((dayofweek(col("incident_date_utc")) + 5) % 7).cast("integer"))
    
    # 2. Đổi thời gian sang giây (Cách mới tối ưu hơn của Spark 3.x)
    df = df.withColumn("ts_seconds", col("incident_date_utc").cast("long"))
    
    # Ép cột 'arrest' (boolean) thành số nguyên (0, 1)
    df = df.withColumn("arrest_int", when(col("arrest") == True, 1).otherwise(0))
    
    days_7_sec = 7 * 24 * 60 * 60
    days_30_sec = 30 * 24 * 60 * 60
    
    # FIX LỖI DATA LEAKAGE: Thay số 0 thành -1 để không đếm vụ án hiện tại
    window_7d = Window.partitionBy("h3_index").orderBy("ts_seconds").rangeBetween(-days_7_sec, -1)
    window_30d = Window.partitionBy("h3_index").orderBy("ts_seconds").rangeBetween(-days_30_sec, -1)
    
    # 3. Tính toán Rolling Features
    # Dùng coalesce(..., lit(0)) để lấp đầy số 0 nếu khu vực đó chưa có tội phạm nào trước đó
    df = df.withColumn("crime_density_7d", coalesce(count("unique_key").over(window_7d), lit(0)).cast("float")) \
           .withColumn("crime_density_30d", coalesce(count("unique_key").over(window_30d), lit(0)).cast("float")) \
           .withColumn("arrest_rate", coalesce(avg("arrest_int").over(window_30d), lit(0.0)).cast("float"))
           
    # Dọn dẹp
    return df.drop("ts_seconds", "arrest_int")