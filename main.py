from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
import cv2
import numpy as np
import onnxruntime as ort
from inference import get_model
import supervision as sv
from typing import Optional, Tuple, Dict, Any
import time
import sys

# 配置管理
class Config:
    MODEL_ID = "taiwan-license-plate-recognition-research-tlprr-rbife/1"
    PREFERRED_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    HOST = "0.0.0.0"
    PORT = 5000
    DEBUG = True

# 模型管理
class ModelManager:
    _instance = None
    
    def __init__(self):
        self.model = None
        self.provider = self._get_provider()
        
    @staticmethod
    def _get_provider() -> str:
        available_providers = ort.get_available_providers()
        print(f"可用的推論提供者: {available_providers}")
        provider = next((p for p in Config.PREFERRED_PROVIDERS 
                        if p in available_providers), "CPUExecutionProvider")
        print(f"使用推論提供者: {provider}")
        return provider
        
    def get_model(self):
        if self.model is None:
            self.model = get_model(model_id=Config.MODEL_ID)
            if hasattr(self.model, "set_providers"):
                self.model.set_providers([self.provider])
        return self.model
    
    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

# 圖片處理
class ImageProcessor:
    @staticmethod
    def decode_base64(base64_string: str) -> Optional[np.ndarray]:
        try:
            image_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(image_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            return None
            
    @staticmethod
    def encode_base64(image: np.ndarray) -> Optional[str]:
        try:
            _, buffer = cv2.imencode('.jpg', image)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception:
            return None
            
    @staticmethod
    def draw_detections(image: np.ndarray, detections: sv.Detections) -> np.ndarray:
        bounding_box_annotator = sv.BoxAnnotator()
        return bounding_box_annotator.annotate(scene=image, detections=detections)

    @staticmethod
    def crop_detection(image: np.ndarray, detection) -> np.ndarray:
        """裁剪檢測框區域"""
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        return image[y1:y2, x1:x2]

# API 處理
class APIHandler:
    def __init__(self):
        self.model_manager = ModelManager.instance()
        self.image_processor = ImageProcessor()
    
    def process_inference(self, base64_image: str) -> Tuple[Dict[str, Any], int]:
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
        """處理推論並回傳裁剪後的檢測區域"""
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
            
            # 裁剪每個檢測區域
            cropped_images = []
            for i in range(len(detections)):
                cropped = self.image_processor.crop_detection(image, detections[i])
                encoded = self.image_processor.encode_base64(cropped)
                if encoded:
                    cropped_images.append({
                        "id": i,
                        "image": encoded
                    })
                
            return {
                "success": True,
                "detections": cropped_images
            }, 200
            
        except Exception as e:
            return {"error": str(e)}, 500

# Flask 應用程式設定
app = Flask(__name__)
CORS(app)
api_handler = APIHandler()

# 預先載入模型
print("正在預載入模型...")
try:
    start_time = time.time()
    model = api_handler.model_manager.get_model()
    load_time = time.time() - start_time
    print(f"模型載入成功！耗時: {load_time:.2f} 秒")
except Exception as e:
    print(f"模型載入失敗: {str(e)}")
    sys.exit(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/infer', methods=['POST'])
def infer():
    if 'image' not in request.json:
        return jsonify({"error": "未提供圖片"}), 400
        
    response, status_code = api_handler.process_inference(request.json['image'])
    return jsonify(response), status_code

@app.route('/infer/crops', methods=['POST'])
def infer_crops():
    """新的 API 端點，只回傳檢測框內的圖片"""
    if 'image' not in request.json:
        return jsonify({"error": "未提供圖片"}), 400
        
    response, status_code = api_handler.process_inference_crops(request.json['image'])
    return jsonify(response), status_code

if __name__ == '__main__':
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
