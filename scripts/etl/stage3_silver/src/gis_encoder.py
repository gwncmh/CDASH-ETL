import h3
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, pandas_udf, lit, acos, cos, sin, radians
from pyspark.sql.types import StringType
from utils.logger import get_logger

logger = get_logger("Stage3_GISEncoder")

CHICAGO_CENTER_LAT = 41.8832
CHICAGO_CENTER_LON = -87.6324
EARTH_RADIUS_KM = 6371.0
H3_RESOLUTION = 8

# KHAI BÁO PANDAS UDF (Xử lý Vectorized siêu tốc bằng Arrow)
@pandas_udf(StringType())
def h3_pandas_udf(lat_series: pd.Series, lon_series: pd.Series) -> pd.Series:
    result = []
    for lat, lon in zip(lat_series, lon_series):
        if pd.isna(lat) or pd.isna(lon):
            result.append(None)
        else:
            # Ép cứng kiểu float đề phòng dữ liệu bẩn
            result.append(h3.latlng_to_cell(float(lat), float(lon), H3_RESOLUTION))
    return pd.Series(result)

def encode_spatial_features(df: DataFrame) -> DataFrame:
    logger.info("GIS Encoder: Tính H3 bằng Pandas UDF và tính Haversine...")
    
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return df

    # 1. Mã hóa H3 bằng Pandas UDF
    df_encoded = df.withColumn("h3_index", h3_pandas_udf(col("latitude"), col("longitude")))
    
    # 2. Tính Haversine
    center_lat_rad = radians(lit(CHICAGO_CENTER_LAT))
    center_lon_rad = radians(lit(CHICAGO_CENTER_LON))
    crime_lat_rad = radians(col("latitude"))
    crime_lon_rad = radians(col("longitude"))
    
    distance_col = lit(EARTH_RADIUS_KM) * acos(
        sin(center_lat_rad) * sin(crime_lat_rad) + 
        cos(center_lat_rad) * cos(crime_lat_rad) * cos(crime_lon_rad - center_lon_rad)
    )
    
    df_encoded = df_encoded.withColumn("distance_to_center_km", distance_col)
    logger.info("GIS Encoder: Hoàn tất!")
    return df_encoded