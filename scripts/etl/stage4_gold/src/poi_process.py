import h3
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, min, when, pandas_udf
from pyspark.sql.types import StringType

# ĐỒNG BỘ ĐỘ PHÂN GIẢI VỚI STAGE 3 (Bắt buộc)
H3_RESOLUTION = 8  

# Sử dụng Vectorized UDF (Arrow) giống hệt Stage 3 để tối ưu tốc độ
@pandas_udf(StringType())
def h3_pandas_udf(lat_series: pd.Series, lon_series: pd.Series) -> pd.Series:
    result = []
    for lat, lon in zip(lat_series, lon_series):
        if pd.isna(lat) or pd.isna(lon):
            result.append(None)
        else:
            try:
                # Dùng latlng_to_cell thay vì geo_to_h3 (hàm cũ)
                result.append(h3.latlng_to_cell(float(lat), float(lon), H3_RESOLUTION))
            except:
                result.append(None)
    return pd.Series(result)

def process_poi_data(df_crime: DataFrame, df_poi_raw: DataFrame) -> DataFrame:
    print("⚡ Đang băm không gian đồng thời 4 loại POI (School, Station, Park, Nightlife)...")
    
    # 1. Trích xuất danh sách H3 duy nhất từ Fact (Đã ở phân giải 9)
    df_unique_h3 = df_crime.filter(col("h3_index").isNotNull()) \
        .select("h3_index").dropDuplicates(["h3_index"])
        
    # 2. Định vị POI lên lưới H3 bằng Pandas UDF siêu tốc
    df_poi_hashed = df_poi_raw \
        .filter(col("latitude").isNotNull() & col("longitude").isNotNull()) \
        .withColumn("poi_h3", h3_pandas_udf(col("latitude"), col("longitude"))) \
        .filter(col("poi_h3").isNotNull())
        
    # 3. SCAN ONCE, AGGREGATE EVERYWHERE
    df_poi_grouped = df_poi_hashed.groupBy("poi_h3").agg(
        min(when(col("poi_type") == "school", 0.2)).alias("dist_school"),
        min(when(col("poi_type") == "transit_station", 0.2)).alias("dist_station"),
        min(when(col("poi_type") == "park", 0.2)).alias("dist_park"),
        min(when(col("poi_type") == "nightlife", 0.2)).alias("dist_nightlife")
    ).withColumnRenamed("poi_h3", "h3_index")
    
    # 4. HASH JOIN
    df_poi_mapping = df_unique_h3.join(df_poi_grouped, on="h3_index", how="left")
    
    # 5. XỬ LÝ DỮ LIỆU KHUYẾT (Imputation)
    df_poi_mapping = df_poi_mapping.na.fill({
        "dist_school": 1.5,
        "dist_station": 2.0,
        "dist_park": 1.2,
        "dist_nightlife": 2.5
    }).select(
        "h3_index",
        col("dist_school").alias("dist_nearest_school"),
        col("dist_station").alias("dist_nearest_station"),
        col("dist_park").alias("dist_nearest_park"),
        col("dist_nightlife").alias("dist_nearest_nightlife")
    )

    print("✅ Đã hoàn thành cấu trúc bảng tra cứu đa đặc trưng POI thành công.")
    return df_poi_mapping