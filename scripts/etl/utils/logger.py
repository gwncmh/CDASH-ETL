import logging
import sys

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    
    # Đảm bảo log của mình không bị đẩy ngược lên logger cha (Spark/Root)
    logger.propagate = False
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Ghi INFO ra stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Chỉ cho phép handler này xử lý log INFO trở xuống
        console_handler.addFilter(lambda record: record.levelno <= logging.INFO)
        
        # Ghi ERROR ra stderr để Cloud Logging bôi đỏ tự động
        error_handler = logging.StreamHandler(sys.stderr)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(error_handler)
        
    return logger