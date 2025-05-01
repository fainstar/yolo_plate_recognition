from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
import cv2
import numpy as np
import onnxruntime as ort
from inference import get_model
import supervision as sv

app = Flask(__name__)
CORS(app)  # 啟用 CORS 支持

# 檢查可用的推論提供者
available_providers = ort.get_available_providers()
print(f"可用的推論提供者: {available_providers}")

# 優先選擇 CUDA，否則使用 CPU
preferred_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
provider = next((p for p in preferred_providers if p in available_providers), "CPUExecutionProvider")
print(f"使用推論提供者: {provider}")

# 全域變數儲存模型，避免重複載入
model = None

def load_model():
    global model
    if model is None:
        model = get_model(model_id="taiwan-license-plate-recognition-research-tlprr-rbife/1")
        if hasattr(model, "set_providers"):
            model.set_providers([provider])
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/infer', methods=['POST'])
def infer():
    try:
        # 檢查請求中是否包含圖片
        if 'image' not in request.json:
            return jsonify({"error": "未提供圖片"}), 400

        # 解碼 base64 圖片
        image_data = base64.b64decode(request.json['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "圖片解碼失敗"}), 400

        # 載入模型（如果尚未載入）
        model = load_model()

        # 執行推論
        results = model.infer(image)[0]

        # 繪製檢測框
        detections = sv.Detections.from_inference(results)
        bounding_box_annotator = sv.BoxAnnotator()
        annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)

        # 將標記後的圖片編碼為 base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        response_image = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "success": True,
            "image": response_image
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
