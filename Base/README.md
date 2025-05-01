# OpenCV YOLOv8 Inference Project

## 專案簡介
本專案使用 YOLOv8 模型進行物件偵測，並透過 CUDA 提供者加速推論。程式會讀取指定的圖像，執行推論，並將結果以標註的方式顯示。

## 專案結構
- `main.py`：主程式，負責加載模型、執行推論並顯示結果。
- `data03.jpeg`：範例圖像，用於測試推論。

## 使用方法
1. 確保已安裝必要的 Python 環境與依賴項。
2. 執行以下命令以運行程式：
   ```bash
   python main.py
   ```
3. 程式將讀取 `data03.jpeg`，執行推論並顯示標註後的圖像。

## 環境需求
- 作業系統：Windows
- Python 版本：3.10
- 依賴項：請參考 `environment.txt`。

## 注意事項
- 確保 CUDA 驅動程式與 `onnxruntime-gpu` 版本相容。
- 如果 CUDA 不可用，程式將自動回退到 CPU 提供者。