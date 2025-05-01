from inference import get_model
import supervision as sv
import cv2
import onnxruntime as ort
import time

# 檢查可用的推論提供者
available_providers = ort.get_available_providers()
print(f"Available providers: {available_providers}")

# 優先選擇 CUDA，否則使用 CPU
preferred_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
provider = next((p for p in preferred_providers if p in available_providers), "CPUExecutionProvider")
print(f"Using provider: {provider}")

# 定義圖片
image_file = "Data/data03.jpeg"
image = cv2.imread(image_file)

try:
    # 模型載入時間
    t0 = time.time()
    model = get_model(model_id="taiwan-license-plate-recognition-research-tlprr-rbife/1")
    t1 = time.time()
    print(f"模型載入耗時: {t1 - t0:.2f} 秒")

    # 設定 provider
    if hasattr(model, "set_providers"):
        model.set_providers([provider])
    t2 = time.time()
    print(f"Provider 設定耗時: {t2 - t1:.2f} 秒")

    # 推論時間
    t3 = time.time()
    results = model.infer(image)[0]
    t4 = time.time()
    print(f"推論耗時: {t4 - t3:.2f} 秒")

    # 畫框時間
    t5 = time.time()
    detections = sv.Detections.from_inference(results)
    bounding_box_annotator = sv.BoxAnnotator()
    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    sv.plot_image(annotated_image)
    t6 = time.time()
    print(f"畫框與顯示耗時: {t6 - t5:.2f} 秒")

    # 總時間
    print(f"總共耗時: {t6 - t0:.2f} 秒")

except Exception as e:
    print(f"An error occurred: {e}")
