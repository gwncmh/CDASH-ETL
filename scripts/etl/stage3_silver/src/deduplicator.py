from pyspark.sql import DataFrame, SparkSession
from utils.logger import get_logger

logger = get_logger("Stage3_Deduplicator")

def remove_existing_records(spark: SparkSession, df_new: DataFrame, target_path: str) -> DataFrame:
    """Loại bỏ dữ liệu đã tồn tại ở Silver Layer bằng kỹ thuật Left Anti Join"""
    logger.info("Deduplicator: Đang kiểm tra dữ liệu trùng lặp (Anti-join)...")
    
    if "unique_key" not in df_new.columns:
        logger.warning("Không tìm thấy cột 'unique_key', hệ thống bỏ qua bước Deduplication.")
        return df_new
        
    try:
        # Cố gắng đọc thư mục đích. Nếu lỗi, nghĩa là chạy lần đầu (First Load)
        try:
            df_existing = spark.read.parquet(target_path).select("unique_key")
            existing_count = df_existing.count()
        except Exception:
            logger.info("Deduplicator: Thư mục đích chưa tồn tại. Đang chạy ở chế độ First Load.")
            return df_new
        
        if existing_count == 0:
            return df_new
            
        logger.info(f"Tìm thấy {existing_count} bản ghi cũ ở Silver. Bắt đầu Anti-Join...")
        
        # Lấy dữ liệu MỚI (chưa từng xuất hiện ở hệ thống cũ)
        df_incremental = df_new.join(df_existing, on="unique_key", how="left_anti")
        
        new_count = df_incremental.count()
        logger.info(f"Deduplicator: Đã lọc xong! Giữ lại {new_count} bản ghi MỚI TINH để xử lý.")
        
        return df_incremental
        
    except Exception as e:
        logger.warning(f"Cảnh báo Deduplicator: {e}. Hệ thống sẽ xử lý toàn bộ file mới.")
        return df_new