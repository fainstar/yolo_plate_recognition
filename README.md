# 車牌辨識系統

## 專案簡介
本專案是一個基於 YOLOv8 模型的車牌辨識系統，提供 Web 介面和 API 服務。系統整合了車牌偵測和 PaddleOCR 文字辨識功能，使用純 CPU 運算以確保最大的相容性。系統使用 base64 進行圖片傳輸，確保資料傳輸的安全性和效率。

## 線上服務
- **Demo 網站：** [https://plat.iside.space/](https://plat.iside.space/)
- **API 端點：** `https://plat.iside.space/infer`

## 系統需求
- Python 3.10 或以上
- Docker（用於容器化部署）
- 建議 2GB 以上記憶體 (經優化後，需求降低)

## 工作目錄結構
```
.
├── Dockerfile              # Docker 配置文件
├── docker-compose.yml      # Docker Compose 配置文件 (可選)
├── main.py                 # 主程式入口
├── README.md               # 專案說明文件
├── requirements.txt        # Python 依賴列表
├── config/                 # 配置管理
│   ├── __init__.py
│   └── config.py          # 系統配置檔案
├── models/                 # 模型管理
│   ├── __init__.py
│   ├── best.pt            # YOLOv8 訓練模型
│   └── model_manager.py   # 模型管理器
├── services/              # 服務層
│   ├── __init__.py
│   └── inference_service.py # 推論服務
├── utils/                 # 工具類
│   ├── __init__.py
│   └── image_processor.py # 圖片處理工具
├── templates/             # 網頁模板
│   ├── index.html        # 主頁面
│   └── process.html      # 處理流程頁面
├── Test/                 # 測試目錄 (部分測試檔案可能未列出)
│   ├── test_base.py
│   └── test_plate_recognition.py
├── Data/                 # 測試資料
│   ├── base_01.png
│   ├── data02.jpg
│   └── data03.jpeg
└── OutPut/              # 輸出結果 (本地測試時)
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
   - 優化的 CPU 運算 (透過 Docker 多階段建置進一步減小映像檔大小)
   - 批次處理能力

3. **友善使用介面**
   - 直覺式網頁界面
   - 即時視覺化結果
   - 拖放式圖片上傳
   - 詳細辨識資訊顯示
   - **自動清理圖片快取**：確保每次上傳都是全新處理，並在離開處理頁面時自動清除。
   - **API 請求重試機制**：提高網路不穩定時的處理成功率。

## Docker 部署

### 1. 建置映像檔 (可選，推薦直接拉取)
如果您想自行建置映像檔：
```bash
# 進入專案根目錄
cd /path/to/your/OpenCV_project
# 建置 Docker 映像檔
docker build -t oomaybeoo/plate-recognition:latest .
```

### 2. 使用 Docker Hub (推薦)
```bash
# 直接從 Docker Hub 拉取最新映像檔
docker pull oomaybeoo/plate-recognition:latest

# 執行容器
docker run -d \
  --name plate-recognition \
  -p 5000:5000 \
  # 如果您需要在容器外存取處理後的圖片，可以掛載 OutPut 目錄
  # -v /your/local/output_path:/app/OutPut \
  # 如果您有固定的測試圖片集，可以掛載 Data 目錄 (唯讀)
  # -v /your/local/data_path:/app/Data:ro \
  oomaybeoo/plate-recognition:latest
```
**注意：** 由於前端現在直接處理 Base64 圖片，`OutPut` 和 `Data` 目錄的掛載主要用於本地開發或特定測試需求。對於標準的 Web 服務使用，通常不需要掛載這些目錄。

### 3. 推送映像檔到 Docker Hub (如果您自行建置並想分享)
```bash
# 登入 Docker Hub
docker login
# 推送映像檔
docker push oomaybeoo/plate-recognition:latest
```

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
            "bbox": [x1, y1, x2, y2],
            "plate_number": "ABC-1234",
            "processing_images": {
                "plate": "base64_encoded_crop",
                "gray": "base64_encoded_gray",
                "binary": "base64_encoded_binary"
            }
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

## 系統配置
在 `config.py` 中可調整的關鍵參數：

```python
# PaddleOCR 配置
PADDLE_LANG = "en"             # 識別語言
PADDLE_MIN_SCORE = 0.3         # 檢測閾值
PADDLE_BOX_THRESH = 0.3        # 框選閾值

# 圖像處理配置
IMAGE_RESIZE_FACTOR = 3.0      # 放大倍數
CONTRAST_ALPHA = 1.5           # 對比度調整
CONTRAST_BETA = 8              # 亮度調整
DENOISE_H = 8                  # 去噪強度
```

## 性能優化建議

### 1. 低解析度圖片處理
- 調高 `IMAGE_RESIZE_FACTOR` 值
- 降低 `DENOISE_H` 以保留更多細節
- 適當調整 `CONTRAST_ALPHA` 和 `CONTRAST_BETA`

### 2. CPU 效能優化
- 調整容器 CPU 核心數限制
- 優化圖片預處理參數
- 批次處理時注意記憶體使用

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

2. **系統效能問題**
   - 檢查 CPU 使用率
   - 監控記憶體使用狀況
   - 調整容器資源限制

3. **辨識準確度問題**
   - 調整預處理參數
   - 檢查圖片品質
   - 優化 OCR 配置

## 版本歷程

### 2025/05/08
- **前端優化**：
    - 新增圖片上傳前自動清理先前結果的功能。
    - 新增處理流程頁面離開時自動清理 sessionStorage 的功能。
    - 為 API 請求 (`/infer` 和 `/infer/crops`) 新增重試機制，提高傳輸穩定性。
    - 調整 API 呼叫順序，確保 `/infer` 成功後才呼叫 `/infer/crops`。
- **Docker 優化**：
    - 大幅精簡 `requirements.txt`，移除不必要的開發和 GPU 相關套件。
    - `Dockerfile` 改為多階段建置，顯著減小最終映像檔大小。
    - 明確在 `Dockerfile` 中安裝 PyTorch CPU 版本。
    - 優化 `Dockerfile` 中的 `COPY` 指令，僅複製必要檔案。
- **文件更新**：更新 README.md 以反映最新的 Docker 指令和前端功能。

### 2025/05/04
- 移除 GPU 相關依賴
- 優化 CPU 運算效能
- 更新 Docker 配置
- 簡化部署流程

### 2025/05/01
- 更新 PaddleOCR 配置
- 優化圖像處理流程
- 改進低解析度圖片處理
- 完善文件說明

## 授權說明
本專案採用 MIT 授權條款。詳見 LICENSE 文件。