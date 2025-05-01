"""推論服務模組"""
from typing import Tuple, Dict, Any
import supervision as sv
from models.model_manager import ModelManager
from utils.image_processor import ImageProcessor
import cv2
import numpy as np
import base64

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
            
            # 使用 supervision 處理檢測結果
            detections = sv.Detections.from_inference(results)
            
            # 處理每個檢測到的區域
            processed_detections = []
            for i in range(len(detections)):
                # 獲取邊界框座標
                xyxy = detections.xyxy[i]  # 直接從 detections 物件獲取座標
                x1, y1, x2, y2 = map(int, xyxy)
                
                # 裁切車牌區域
                cropped = image[y1:y2, x1:x2]
                
                # 預處理步驟
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # OCR 辨識
                plate_text = self.image_processor.recognize_text(cropped)
                
                processed_detections.append({
                    'id': i,
                    'bbox': xyxy.tolist(),
                    'plate_number': plate_text,
                    'processing_images': {
                        'plate': self.image_processor.encode_base64(cropped),
                        'gray': self.image_processor.encode_base64(gray),
                        'binary': self.image_processor.encode_base64(binary)
                    }
                })
                
            return {
                'success': True,
                'detections': processed_detections
            }, 200
            
        except Exception as e:
            print(f"處理錯誤: {str(e)}")  # 添加錯誤日誌
            return {'success': False, 'error': str(e)}, 500

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

    def process_image(self, image_data):
        try:
            # 解碼並讀取圖片
            img_array = np.frombuffer(base64.b64decode(image_data), np.uint8)
            original_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            processing_steps = {
                'original': original_image.copy(),
            }

            # 偵測車牌
            detect_result = self.image_processor.detect_plates(original_image)
            detection_image = detect_result['image']
            processing_steps['detection'] = detection_image.copy()

            plates_info = []
            for i, (plate_img, bbox) in enumerate(detect_result['plates']):
                # 儲存裁切的車牌
                processing_steps[f'plate_{i}'] = plate_img.copy()
                
                # 預處理步驟
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                processing_steps[f'gray_{i}'] = gray.copy()
                
                # 二值化
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                processing_steps[f'binary_{i}'] = binary.copy()
                
                # 其他預處理步驟...
                # 可以根據需要添加更多處理步驟

                # 模擬車牌辨識結果
                plate_number = "ABC-1234"  # 這裡應該是實際的辨識結果
                
                plates_info.append({
                    'id': i,
                    'bbox': bbox.tolist(),
                    'plate_number': plate_number,
                    'processing_images': {
                        'plate': self._encode_image(plate_img),
                        'gray': self._encode_image(gray),
                        'binary': self._encode_image(binary)
                    }
                })

            # 編碼所有處理步驟的圖片
            encoded_steps = {}
            for step_name, step_image in processing_steps.items():
                encoded_steps[step_name] = self._encode_image(step_image)

            return {
                'success': True,
                'steps': encoded_steps,
                'plates': plates_info,
                'image': self._encode_image(detection_image)
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _encode_image(self, image):
        """將圖片編碼為 base64 字串"""
        success, encoded_image = cv2.imencode('.jpg', image)
        if success:
            return base64.b64encode(encoded_image.tobytes()).decode('utf-8')
        return None