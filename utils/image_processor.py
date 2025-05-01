"""圖片處理模組"""
import base64
import cv2
import numpy as np
import supervision as sv
from typing import Optional, List, Dict
from paddleocr import PaddleOCR
from config.config import config

# 初始化 PaddleOCR
ocr = PaddleOCR(
    use_angle_cls=config.PADDLE_USE_ANGLE_CLS,
    lang=config.PADDLE_LANG,
    use_gpu=config.PADDLE_USE_GPU,
    det_db_score_mode="slow",  # 使用更準確的檢測模式
    det_db_box_thresh=config.PADDLE_BOX_THRESH,
    det_db_thresh=config.PADDLE_MIN_SCORE,
    det_db_unclip_ratio=config.PADDLE_UNCLIP_RATIO,
    show_log=False
)

class ImageProcessor:
    """圖片處理類"""
    
    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """增強圖片品質"""
        # 放大圖片
        height, width = image.shape[:2]
        new_height = int(height * config.IMAGE_RESIZE_FACTOR)
        new_width = int(width * config.IMAGE_RESIZE_FACTOR)
        resized = cv2.resize(image, (new_width, new_height), 
                           interpolation=cv2.INTER_CUBIC)
        
        # 去噪處理
        denoised = cv2.fastNlMeansDenoisingColored(
            resized,
            None,
            h=config.DENOISE_H,
            hColor=config.DENOISE_H,
            templateWindowSize=config.DENOISE_TEMPLATE_WINDOW,
            searchWindowSize=config.DENOISE_SEARCH_WINDOW
        )
        
        # 增強對比度和亮度
        enhanced = cv2.convertScaleAbs(
            denoised,
            alpha=config.CONTRAST_ALPHA,
            beta=config.CONTRAST_BETA
        )
        
        # 銳化處理
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    
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
        # 增強圖片品質
        enhanced = ImageProcessor.enhance_image(image)
        
        # 轉換為灰度圖
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
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
    def clean_plate_text(text: str) -> str:
        """清理車牌文字"""
        # 只保留英文字母、數字和破折號
        cleaned = ''.join(c for c in text if c.isalnum() or c == '-')
        return cleaned.upper()

    @staticmethod
    def recognize_text(image: np.ndarray) -> str:
        """執行 OCR 文字辨識"""
        try:
            # 預處理圖片
            processed_image = ImageProcessor.preprocess_for_ocr(image)
            
            # 使用 PaddleOCR 執行辨識
            result = ocr.ocr(processed_image, cls=False)
            
            if result and result[0]:
                # 提取最高信心度的文字
                texts = []
                for line in result[0]:
                    text = line[1][0]  # 獲取識別的文字
                    confidence = line[1][1]  # 獲取信心度
                    texts.append((text, confidence))
                
                # 按信心度排序並取最高的
                texts.sort(key=lambda x: x[1], reverse=True)
                if texts:
                    # 清理並回傳最高信心度的文字
                    return ImageProcessor.clean_plate_text(texts[0][0])
            
            return ""
            
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
            # 增強裁剪後的圖片
            enhanced_crop = cls.enhance_image(cropped)
            encoded = cls.encode_base64(enhanced_crop)
            # 執行 OCR
            plate_text = cls.recognize_text(enhanced_crop)
            if encoded:
                cropped_images.append({
                    "id": i,
                    "image": encoded,
                    "plate_number": plate_text
                })
        return cropped_images