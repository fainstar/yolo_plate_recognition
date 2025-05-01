"""配置管理模組"""
from dataclasses import dataclass

@dataclass
class AppConfig:
    """應用程式配置"""
    MODEL_ID: str = "taiwan-license-plate-recognition-research-tlprr-rbife/1"
    PREFERRED_PROVIDERS: list = None
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    DEBUG: bool = True
    
    # PaddleOCR 配置
    PADDLE_USE_GPU: bool = True
    PADDLE_LANG: str = "en"  # 使用英文識別
    PADDLE_USE_ANGLE_CLS: bool = True  # 使用文字方向分類
    PADDLE_MIN_SCORE: float = 0.3  # 降低文字檢測的信心度閾值
    PADDLE_BOX_THRESH: float = 0.3  # 降低檢測框的信心度閾值
    PADDLE_UNCLIP_RATIO: float = 2.0  # 擴大檢測框範圍
    
    # 圖像預處理配置
    PREPROCESSING_THRESHOLD: int = 128  # 二值化閾值
    PREPROCESSING_GAUSSIAN_KERNEL: tuple = (3, 3)  # 降低高斯模糊核大小以保留更多細節
    PREPROCESSING_GAUSSIAN_SIGMA: int = 1  # 適度的高斯模糊
    
    # 圖像增強配置
    IMAGE_RESIZE_FACTOR: float = 3.0  # 放大倍數
    CONTRAST_ALPHA: float = 1.5  # 對比度增強係數
    CONTRAST_BETA: int = 8  # 亮度調整值
    SHARPEN_KERNEL: tuple = (3, 3)  # 銳化核大小
    SHARPEN_SIGMA: float = 1.0  # 銳化強度
    DENOISE_H: int = 8  # 去噪強度
    DENOISE_TEMPLATE_WINDOW: int = 7  # 去噪模板窗口大小
    DENOISE_SEARCH_WINDOW: int = 21  # 去噪搜索窗口大小

    def __post_init__(self):
        if self.PREFERRED_PROVIDERS is None:
            self.PREFERRED_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

config = AppConfig()