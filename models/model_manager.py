"""模型管理模組"""
import onnxruntime as ort
from inference import get_model
from typing import Optional
from config.config import config

class ModelManager:
    """模型管理類"""
    _instance: Optional['ModelManager'] = None
    
    def __init__(self):
        self.model = None
        self.provider = self._get_provider()
        
    @staticmethod
    def _get_provider() -> str:
        """獲取可用的推論提供者"""
        available_providers = ort.get_available_providers()
        print(f"可用的推論提供者: {available_providers}")
        provider = next((p for p in config.PREFERRED_PROVIDERS 
                        if p in available_providers), "CPUExecutionProvider")
        print(f"使用推論提供者: {provider}")
        return provider
        
    def get_model(self):
        """獲取模型實例"""
        if self.model is None:
            self.model = get_model(model_id=config.MODEL_ID)
            if hasattr(self.model, "set_providers"):
                self.model.set_providers([self.provider])
        return self.model
    
    @classmethod
    def instance(cls) -> 'ModelManager':
        """獲取單例實例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance