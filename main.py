"""主程式入口"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import time
import sys
from services.inference_service import InferenceService
from models.model_manager import ModelManager
from config.config import config

# Flask 應用程式設定
app = Flask(__name__)
CORS(app)

# 初始化服務
inference_service = InferenceService()

# 預先載入模型
print("正在預載入模型...")
try:
    start_time = time.time()
    model = ModelManager.instance().get_model()
    load_time = time.time() - start_time
    print(f"模型載入成功！耗時: {load_time:.2f} 秒")
except Exception as e:
    print(f"模型載入失敗: {str(e)}")
    sys.exit(1)

@app.route('/')
def index():
    """首頁路由"""
    return render_template('index.html')

@app.route('/infer', methods=['POST'])
def infer():
    """推論 API 路由"""
    if 'image' not in request.json:
        return jsonify({"error": "未提供圖片"}), 400
        
    response, status_code = inference_service.process_inference(request.json['image'])
    return jsonify(response), status_code

@app.route('/infer/crops', methods=['POST'])
def infer_crops():
    """車牌裁剪 API 路由"""
    if 'image' not in request.json:
        return jsonify({"error": "未提供圖片"}), 400
        
    response, status_code = inference_service.process_inference_crops(request.json['image'])
    return jsonify(response), status_code

@app.route('/infer/ocr', methods=['POST'])
def infer_ocr():
    """車牌 OCR API 路由"""
    if 'image' not in request.json:
        return jsonify({"error": "未提供圖片"}), 400
        
    response, status_code = inference_service.process_plate_ocr(request.json['image'])
    return jsonify(response), status_code

if __name__ == '__main__':
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )
