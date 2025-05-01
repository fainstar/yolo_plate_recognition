"""圖片處理模組"""
import base64
import cv2
import numpy as np
import supervision as sv
from typing import Optional, List, Dict
import pytesseract
from config.config import config

# 設定 Tesseract 執行檔路徑
pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

class ImageProcessor:
    """圖片處理類"""
    
    @staticmethod
    def decode_base64(base64_string: str) -> Optional[np.ndarray]:
        """解碼 base64 圖片"""
        try:
            image_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(image_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            return None
            
    @staticmethod
    def encode_base64(image: np.ndarray) -> Optional[str]:
        """編碼圖片為 base64"""
        try:
            _, buffer = cv2.imencode('.jpg', image)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception:
            return None
            
    @staticmethod
    def draw_detections(image: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """在圖片上繪製檢測框"""
        bounding_box_annotator = sv.BoxAnnotator()
        return bounding_box_annotator.annotate(scene=image, detections=detections)

    @staticmethod
    def crop_detection(image: np.ndarray, detection) -> np.ndarray:
        """裁剪檢測框區域"""
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        return image[y1:y2, x1:x2]

    @staticmethod
    def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
        """預處理圖片以提高 OCR 準確度"""
        # 轉換為灰度圖
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(
            gray, 
            config.PREPROCESSING_GAUSSIAN_KERNEL,
            config.PREPROCESSING_GAUSSIAN_SIGMA
        )
        
        # 自適應二值化
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # 鄰域大小
            2    # 常數減項
        )
        
        # 執行形態學操作以移除雜訊
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return morph

    @staticmethod
    def recognize_text(image: np.ndarray) -> str:
        """執行 OCR 文字辨識"""
        try:
            # 預處理圖片
            processed_image = ImageProcessor.preprocess_for_ocr(image)
            
            # 使用配置的 whitelist 執行 OCR
            text = pytesseract.image_to_string(
                processed_image,
                lang=config.TESSERACT_LANG,
                config=config.TESSERACT_CONFIG
            )
            
            # 清理結果
            cleaned_text = text.strip().replace(" ", "").upper()
            
            # 驗證結果是否只包含允許的字符
            allowed_chars = set(config.TESSERACT_CHAR_WHITELIST)
            cleaned_text = ''.join(c for c in cleaned_text if c in allowed_chars)
            
            return cleaned_text
            
        except Exception as e:
            print(f"OCR 辨識錯誤: {str(e)}")
            return ""
        
    @classmethod
    def process_detections(cls, image: np.ndarray, 
                         detections: sv.Detections) -> List[Dict]:
        """處理所有檢測結果"""
        cropped_images = []
        for i in range(len(detections)):
            cropped = cls.crop_detection(image, detections[i])
            encoded = cls.encode_base64(cropped)
            # 執行 OCR
            plate_text = cls.recognize_text(cropped)
            if encoded:
                cropped_images.append({
                    "id": i,
                    "image": encoded,
                    "plate_number": plate_text
                })
        return cropped_images