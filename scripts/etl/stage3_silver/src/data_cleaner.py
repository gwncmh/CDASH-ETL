from pyspark.sql import DataFrame
from pyspark.sql.functions import col, year, current_date
from utils.logger import get_logger

logger = get_logger("Stage3_DataCleaner")

# Tọa độ giới hạn (Bounding Box) của thành phố Chicago:
# Chỉ lấy những điểm nằm gọn trong hình chữ nhật này để loại bỏ rác
CHICAGO_BOUNDS = {
    "min_lat": 41.64, "max_lat": 42.02,
    "min_lon": -87.94, "max_lon": -87.52
}

def clean_data(df: DataFrame) -> DataFrame:
    """
    Module: Data Cleaner
    Vai trò: Lọc nhiễu không gian (Spatial), thời gian (Temporal) và xử lý Null.
    """
    logger.info("Data Cleaner: Bắt đầu làm sạch dữ liệu (Filtering & Deduplication)...")
    
    # Lưu lại số lượng dòng ban đầu để thống kê
    initial_count = df.count()
    
    # 1. Loại bỏ các dòng thiếu thông tin cốt lõi (Không khoan nhượng)
    critical_cols = ["unique_key", "case_number", "incident_date_utc", "latitude", "longitude", "community_area"]
    # Đảm bảo cột tồn tại trước khi check null để không bị lỗi
    existing_critical = [c for c in critical_cols if c in df.columns] 
    df_clean = df.na.drop(subset=existing_critical)
    
    # 2. Xóa trùng lặp trong Batch (Sử dụng unique_key là khóa chính)
    if "unique_key" in df_clean.columns:
        df_clean = df_clean.dropDuplicates(["unique_key"])
        
    # 3. Bộ lọc Không gian (Spatial Filtering) - Giới hạn ranh giới Chicago
    if "latitude" in df_clean.columns and "longitude" in df_clean.columns:
        df_clean = df_clean.filter(
            (col("latitude") >= CHICAGO_BOUNDS["min_lat"]) & 
            (col("latitude") <= CHICAGO_BOUNDS["max_lat"]) & 
            (col("longitude") >= CHICAGO_BOUNDS["min_lon"]) & 
            (col("longitude") <= CHICAGO_BOUNDS["max_lon"])
        )
        
    # 4. Bộ lọc Thời gian (Temporal Filtering) - Dữ liệu Chicago Crime bắt đầu từ 2001
    if "incident_date_utc" in df_clean.columns:
        df_clean = df_clean.filter(
            (year(col("incident_date_utc")) >= 2001) & 
            (col("incident_date_utc") <= current_date()) # Không lấy dữ liệu của tương lai
        )
        
    # Tính toán lượng rác bị loại bỏ
    final_count = df_clean.count()
    logger.info(f"Data Cleaner: Hoàn tất. Đã lọc bỏ {initial_count - final_count} bản ghi rác/lỗi.")
    
    return df_clean