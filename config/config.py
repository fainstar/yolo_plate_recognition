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
    
    # Tesseract 配置
    TESSERACT_CMD: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    TESSERACT_LANG: str = "eng"
    # 限制只能辨識英文字母、數字和破折號
    TESSERACT_CHAR_WHITELIST: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
    
    # 圖像預處理配置
    PREPROCESSING_THRESHOLD: int = 128  # 二值化閾值
    PREPROCESSING_GAUSSIAN_KERNEL: tuple = (5, 5)  # 高斯模糊核大小
    PREPROCESSING_GAUSSIAN_SIGMA: int = 0  # 高斯模糊標準差

    def __post_init__(self):
        if self.PREFERRED_PROVIDERS is None:
            self.PREFERRED_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            
    @property
    def TESSERACT_CONFIG(self) -> str:
        """取得 Tesseract 配置字串"""
        return f"-c tessedit_char_whitelist={self.TESSERACT_CHAR_WHITELIST} --psm 7"

config = AppConfig()