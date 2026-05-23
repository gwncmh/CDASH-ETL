# utils/__init__.py

from .logger import get_logger

# Chỉ cho phép import hàm này khi dùng cú pháp: from utils import *
__all__ = ["get_logger"]