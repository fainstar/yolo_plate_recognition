import requests
import base64
import json
import cv2
import numpy as np
from pathlib import Path

def test_plate_recognition():
    """測試車牌辨識 API"""
    print("\n開始測試車牌辨識 API...")
    
    # 讀取測試圖片
    image_path = Path("Data/data02.jpg")
    if not image_path.exists():
        print(f"錯誤：找不到測試圖片 {image_path}")
        return
    
    # 將圖片編碼為 base64
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    try:
        # 發送請求到車牌辨識 API
        response = requests.post(
            'http://localhost:5000/infer/crops',
            json={"image": encoded_string},
            timeout=30
        )
        
        # 檢查回應狀態
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                print(f"\n成功偵測到 {len(data['detections'])} 個車牌")
                
                # 建立輸出目錄
                output_dir = Path("OutPut/plates")
                output_dir.mkdir(exist_ok=True)
                
                # 處理每個偵測結果
                for i, detection in enumerate(data["detections"], 1):
                    # 儲存車牌圖片
                    plate_image = base64.b64decode(detection["image"])
                    output_path = output_dir / f"plate_{i}.jpg"
                    with open(output_path, "wb") as f:
                        f.write(plate_image)
                    
                    # 顯示辨識結果
                    print(f"\n車牌 #{i}:")
                    print(f"- 儲存路徑: {output_path}")
                    print(f"- 辨識號碼: {detection.get('plate_number', '無法辨識')}")
                    
            else:
                print("處理失敗：API 回傳錯誤")
        else:
            print(f"請求失敗：HTTP {response.status_code}")
            print(f"錯誤訊息：{response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"網路請求錯誤：{str(e)}")
    except Exception as e:
        print(f"執行過程發生錯誤：{str(e)}")

if __name__ == "__main__":
    test_plate_recognition()