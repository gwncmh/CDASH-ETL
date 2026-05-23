from pyspark.sql import DataFrame
from pyspark.sql.functions import col, trim, upper

from utils.logger import get_logger

logger = get_logger("Stage3_SchemaNormalizer")

def normalize_schema(df: DataFrame) -> DataFrame:
    """
    Module: Schema Normalizer
    Vai trò: Ép kiểu dữ liệu nghiêm ngặt, chuẩn hóa text và loại bỏ cột rác.
    """
    logger.info("Schema Normalizer: Bắt đầu tinh chỉnh dữ liệu...")
    
    # 1. Xử lý Thời gian (Dữ liệu gốc là ISO-8601 UTC)
    # Vì dữ liệu đã có chữ 'Z' (UTC), ta ép kiểu thẳng, không cần convert từ Chicago nữa
    if "date" in df.columns:
        df = df.withColumn("incident_date_utc", col("date").cast("timestamp"))
        
    # 2. Xóa các cột thừa thãi để tiết kiệm dung lượng lưu trữ
    cols_to_drop = ["location", "x_coordinate", "y_coordinate", "updated_on", "date"]
    existing_drop_cols = [c for c in cols_to_drop if c in df.columns]
    if existing_drop_cols:
        df = df.drop(*existing_drop_cols)

    # 3. Chuẩn hóa chuỗi Text (Uppercase và Trim khoảng trắng 2 đầu)
    text_cols = ["primary_type", "location_description", "description", "block"]
    for c in text_cols:
        if c in df.columns:
            df = df.withColumn(c, upper(trim(col(c))))

    # 4. Ép kiểu dữ liệu khắt khe (Explicit Casting)
    # Boolean
    if "arrest" in df.columns:
        df = df.withColumn("arrest", col("arrest").cast("boolean"))
    if "domestic" in df.columns:
        df = df.withColumn("domestic", col("domestic").cast("boolean"))
        
    # Integer (ID, District, Ward, Community Area)
    int_cols = ["unique_key", "beat", "district", "ward", "community_area"]
    for c in int_cols:
        if c in df.columns:
            df = df.withColumn(c, col(c).cast("integer"))
            
    # Float (Lat/Lon)
    if "latitude" in df.columns:
        df = df.withColumn("latitude", col("latitude").cast("double"))
    if "longitude" in df.columns:
        df = df.withColumn("longitude", col("longitude").cast("double"))

    logger.info("Schema Normalizer: Đã đúc lại cấu trúc (Schema) thành công!")
    return df