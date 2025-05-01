# 車牌辨識系統

## 專案簡介
本專案是一個基於 YOLOv8 模型的車牌辨識系統，提供 Web 介面和 API 服務。系統整合了車牌偵測和 PaddleOCR 文字辨識功能，支援 GPU 加速並自動回退到 CPU 運算。系統使用 base64 進行圖片傳輸，確保資料傳輸的安全性和效率。

## 系統需求
- Python 3.10 或以上
- CUDA 11.6（用於 GPU 加速，可選）
- NVIDIA 顯示卡驅動程式（用於 GPU 加速，可選）

## 工作目錄結構
```
.
├── main.py                 # 主程式入口
├── config/                 # 配置管理
│   ├── __init__.py
│   └── config.py          # 系統配置檔案
├── models/                 # 模型管理
│   ├── __init__.py
│   └── model_manager.py   # 模型管理器
├── services/              # 服務層
│   ├── __init__.py
│   └── inference_service.py # 推論服務
├── utils/                 # 工具類
│   ├── __init__.py
│   └── image_processor.py # 圖片處理工具
├── templates/             # 網頁模板
│   └── index.html        # Web 介面
├── Test/                 # 測試目錄
│   ├── test_api.py       # API 測試
│   ├── test_base.py      # 基礎功能測試
│   └── test_plate_recognition.py # 車牌辨識測試
├── Data/                 # 測試資料
│   ├── base_01.png
│   ├── data02.jpg
│   └── data03.jpeg
└── OutPut/              # 輸出結果
    ├── result.jpg
    └── plates/          # 裁剪的車牌圖片
```

## 特色功能
1. **智慧圖像處理**
   - 自適應圖像增強
   - 智能去噪和銳化
   - 多階段預處理優化
   - 自動亮度和對比度調整

2. **高效能辨識引擎**
   - YOLOv8 車牌檢測
   - PaddleOCR 文字識別
   - GPU 加速支援
   - 批次處理能力

3. **友善使用介面**
   - 直覺式網頁界面
   - 即時視覺化結果
   - 拖放式圖片上傳
   - 詳細辨識資訊顯示

## API 服務說明

### 1. 車牌檢測（/infer）
**請求方式：** POST
```json
{
    "image": "base64_encoded_image_string"
}
```
**回應：**
```json
{
    "success": true,
    "image": "base64_encoded_result_image"
}
```

### 2. 車牌裁剪與辨識（/infer/crops）
**請求方式：** POST
```json
{
    "image": "base64_encoded_image_string"
}
```
**回應：**
```json
{
    "success": true,
    "detections": [
        {
            "id": 0,
            "image": "base64_encoded_crop",
            "plate_number": "ABC-1234"
        }
    ]
}
```

### 3. OCR 辨識（/infer/ocr）
**請求方式：** POST
```json
{
    "image": "base64_encoded_plate_image"
}
```
**回應：**
```json
{
    "success": true,
    "plate_number": "ABC-1234"
}
```

## 安裝配置

### 1. 環境準備
```bash
# 建立虛擬環境
python -m venv .venv

# 啟動虛擬環境
.venv\Scripts\activate

# 安裝依賴套件
pip install -r requirements.txt
```

### 2. 系統配置
在 `config.py` 中可調整的關鍵參數：

```python
# PaddleOCR 配置
PADDLE_USE_GPU = True          # 是否使用 GPU
PADDLE_LANG = "en"             # 識別語言
PADDLE_MIN_SCORE = 0.3         # 檢測閾值
PADDLE_BOX_THRESH = 0.3        # 框選閾值

# 圖像處理配置
IMAGE_RESIZE_FACTOR = 3.0      # 放大倍數
CONTRAST_ALPHA = 1.5           # 對比度調整
CONTRAST_BETA = 10             # 亮度調整
DENOISE_H = 10                 # 去噪強度
```

## 性能優化建議

### 1. 低解析度圖片處理
- 調高 `IMAGE_RESIZE_FACTOR` 值
- 降低 `DENOISE_H` 以保留更多細節
- 適當調整 `CONTRAST_ALPHA` 和 `CONTRAST_BETA`

### 2. GPU 加速優化
- 確保 CUDA 版本相容
- 預先載入模型到 GPU 記憶體
- 批次處理時善用 GPU 並行運算

### 3. API 效能優化
- 使用異步處理大量請求
- 實作請求佇列管理
- 做好錯誤處理和重試機制

## 故障排除

### 常見問題解決方案

1. **圖片無法正確讀取**
   - 檢查圖片格式是否支援
   - 確認 base64 編碼是否完整
   - 檢查圖片檔案權限

2. **GPU 相關問題**
   - 確認 CUDA 驅動程式版本
   - 檢查 GPU 記憶體使用狀況
   - 確認 CUDA_VISIBLE_DEVICES 設置

3. **辨識準確度問題**
   - 調整預處理參數
   - 檢查圖片品質
   - 優化 OCR 配置

## 開發團隊
若有任何問題或建議，請聯絡：
- Email：[您的信箱]
- GitHub：[您的 GitHub]

## 版本歷程

### 2025/05/01
- 更新 PaddleOCR 配置
- 優化圖像處理流程
- 改進低解析度圖片處理
- 完善文件說明

## 授權說明
本專案採用 MIT 授權條款。詳見 LICENSE 文件。