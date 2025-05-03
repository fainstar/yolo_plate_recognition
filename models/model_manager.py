"""模型管理模組"""
from ultralytics import YOLO
from typing import Optional
from config.config import config

class ModelManager:
    """模型管理類"""
    _instance: Optional['ModelManager'] = None
    
    def __init__(self):
        self.model = None
        
    def get_model(self):
        """獲取模型實例"""
        if self.model is None:
            try:
                self.model = YOLO(config.MODEL_PATH)
            except Exception as e:
                print(f"模型載入失敗: {str(e)}")
                raise
        return self.model
    
    @classmethod
    def instance(cls) -> 'ModelManager':
        """獲取單例實例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance