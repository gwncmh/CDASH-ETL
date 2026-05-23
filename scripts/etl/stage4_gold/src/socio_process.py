from pyspark.sql import DataFrame
from pyspark.sql.functions import col, upper, trim

def clean_socio_data(df_socio: DataFrame, df_pop: DataFrame) -> DataFrame:
    print("Đang xử lý và gộp dữ liệu Dân số & Kinh tế...")
    
    # 1. Chuẩn hóa tên khu vực (Kết hợp upper + trim để khử khoảng trắng thừa)
    df_socio_clean = df_socio.withColumn("join_name", upper(trim(col("community_area_name"))))
    df_pop_clean = df_pop.withColumn("join_name", upper(trim(col("Community Area"))))
    
    # 2. Join Dân số vào Kinh tế
    df_dim_socio = df_socio_clean.join(df_pop_clean, on="join_name", how="left")
    
    # 3. Lựa chọn, ép kiểu cột (Quan trọng: Ep ca về Integer để JOIN với bảng Crime)
    df_dim_socio = df_dim_socio.select(
        col("ca").cast("integer").alias("ca_num"), 
        col("percent_aged_16_unemployed").cast("double").alias("unemployment_rate"),
        col("per_capita_income_").cast("double").alias("per_capita_income"),
        col("hardship_index").cast("integer").alias("hardship_index"),
        col("Total Population").cast("double").alias("population")
    )
    
    print("Đã tạo thành công bảng Bối cảnh Xã hội (dim_socio) duy nhất.")
    return df_dim_socio