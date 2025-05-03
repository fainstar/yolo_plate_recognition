"""配置管理模組"""
from dataclasses import dataclass

@dataclass
class AppConfig:
    """應用程式配置"""
    MODEL_PATH: str = "models/best.pt"
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    DEBUG: bool = True
    
    # PaddleOCR 配置
    PADDLE_USE_GPU: bool = False  # 改為 False，使用 CPU
    PADDLE_LANG: str = "en"
    PADDLE_USE_ANGLE_CLS: bool = True
    PADDLE_MIN_SCORE: float = 0.3
    PADDLE_BOX_THRESH: float = 0.3
    PADDLE_UNCLIP_RATIO: float = 2.0
    
    # 圖像預處理配置
    PREPROCESSING_THRESHOLD: int = 128
    PREPROCESSING_GAUSSIAN_KERNEL: tuple = (3, 3)
    PREPROCESSING_GAUSSIAN_SIGMA: int = 1
    
    # 圖像增強配置
    IMAGE_RESIZE_FACTOR: float = 3.0
    CONTRAST_ALPHA: float = 1.5
    CONTRAST_BETA: int = 8
    SHARPEN_KERNEL: tuple = (3, 3)
    SHARPEN_SIGMA: float = 1.0
    DENOISE_H: int = 8
    DENOISE_TEMPLATE_WINDOW: int = 7
    DENOISE_SEARCH_WINDOW: int = 21

config = AppConfig()