"""推論服務模組"""
from typing import Tuple, Dict, Any
import supervision as sv
from models.model_manager import ModelManager
from utils.image_processor import ImageProcessor

class InferenceService:
    """推論服務類"""
    
    def __init__(self):
        self.model_manager = ModelManager.instance()
        self.image_processor = ImageProcessor()
    
    def process_inference(self, base64_image: str) -> Tuple[Dict[str, Any], int]:
        """處理完整推論請求"""
        # 解碼圖片
        image = self.image_processor.decode_base64(base64_image)
        if image is None:
            return {"error": "圖片解碼失敗"}, 400
            
        try:
            # 執行推論
            model = self.model_manager.get_model()
            results = model.infer(image)[0]
            
            # 處理結果
            detections = sv.Detections.from_inference(results)
            annotated_image = self.image_processor.draw_detections(image, detections)
            
            # 編碼結果
            response_image = self.image_processor.encode_base64(annotated_image)
            if response_image is None:
                return {"error": "結果圖片編碼失敗"}, 500
                
            return {
                "success": True,
                "image": response_image
            }, 200
            
        except Exception as e:
            return {"error": str(e)}, 500

    def process_inference_crops(self, base64_image: str) -> Tuple[Dict[str, Any], int]:
        """處理裁剪推論請求"""
        # 解碼圖片
        image = self.image_processor.decode_base64(base64_image)
        if image is None:
            return {"error": "圖片解碼失敗"}, 400
            
        try:
            # 執行推論
            model = self.model_manager.get_model()
            results = model.infer(image)[0]
            
            # 處理結果
            detections = sv.Detections.from_inference(results)
            cropped_images = self.image_processor.process_detections(image, detections)
                
            return {
                "success": True,
                "detections": cropped_images
            }, 200
            
        except Exception as e:
            return {"error": str(e)}, 500

    def process_plate_ocr(self, base64_image: str) -> Tuple[Dict[str, Any], int]:
        """處理單張車牌圖片的 OCR 辨識請求"""
        # 解碼圖片
        image = self.image_processor.decode_base64(base64_image)
        if image is None:
            return {"error": "圖片解碼失敗"}, 400
            
        try:
            # 直接執行 OCR
            plate_text = self.image_processor.recognize_text(image)
            
            return {
                "success": True,
                "plate_number": plate_text
            }, 200
            
        except Exception as e:
            return {"error": str(e)}, 500