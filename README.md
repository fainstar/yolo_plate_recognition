# 車牌辨識系統

## 專案簡介
本專案是一個基於 YOLOv8 模型的車牌辨識系統，提供 Web 介面和 API 服務。系統整合了車牌偵測和 OCR 文字辨識功能，支援 GPU 加速並自動回退到 CPU 運算。系統使用 base64 進行圖片傳輸，確保資料傳輸的安全性和效率。

## 專案結構
```
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
├── Base/                  # 基礎配置
│   └── environment.txt    # 環境依賴
├── Data/                  # 測試資料
│   ├── base_01.png
│   └── data03.jpeg
├── OutPut/               # 輸出目錄
│   └── result.jpg
├── templates/            # 網頁模板
│   └── index.html
└── Test/                 # 測試目錄
    ├── test_api.py
    └── test_base.py
```

## 功能特點
- Web 介面支援拖放上傳圖片
- RESTful API 支援 base64 圖片傳輸
- GPU 加速支援（自動回退到 CPU）
- 即時車牌偵測和標記
- OCR 車牌文字辨識
- 支援批次處理
- 完整的錯誤處理機制
- 模組化設計，易於擴展

## 系統需求
- Python 3.10 或以上
- CUDA 11.6（用於 GPU 加速，可選）
- onnxruntime-gpu 1.15.1
- Tesseract OCR 5.0.0 或以上

## 安裝步驟

### 1. 安裝 Tesseract OCR
1. 下載 Tesseract OCR 安裝程式：
   - 前往 https://github.com/UB-Mannheim/tesseract/wiki
   - 下載 Windows 64-bit 版本
   - 例如：tesseract-ocr-w64-setup-v5.3.1.20230401.exe

2. 執行安裝：
   - 建議路徑：`C:\Program Files\Tesseract-OCR`
   - 勾選「Additional language data」
   - 將安裝路徑加入系統環境變數 PATH

3. 驗證安裝：
   ```bash
   tesseract --version
   ```

### 2. 安裝 Python 套件
```bash
pip install -r Base/environment.txt
```

## API 端點

### 1. 車牌偵測與標記
- 端點：`/infer`
- 方法：POST
- 功能：偵測車牌位置並在原圖上標記
- 回應：標記後的完整圖片

### 2. 車牌裁剪與辨識
- 端點：`/infer/crops`
- 方法：POST
- 功能：偵測車牌、裁剪並進行 OCR 辨識
- 回應：裁剪的車牌圖片和辨識文字

### 3. 單一車牌 OCR
- 端點：`/infer/ocr`
- 方法：POST
- 功能：對單一車牌圖片進行 OCR 辨識
- 回應：辨識出的車牌號碼

## 配置說明

### OCR 配置
```python
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_LANG = "eng"
TESSERACT_CHAR_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
```

### 圖像預處理配置
```python
PREPROCESSING_THRESHOLD = 128
PREPROCESSING_GAUSSIAN_KERNEL = (5, 5)
PREPROCESSING_GAUSSIAN_SIGMA = 0
```

## 效能優化
- 單例模式管理模型載入
- GPU 加速支援
- 圖像預處理優化
- 批次處理支援

## 未來規劃
- [ ] 支援多國語言車牌
- [ ] 整合資料庫儲存辨識結果
- [ ] 提供批次處理 API
- [ ] 添加即時視訊處理支援

## 貢獻指南
1. Fork 專案
2. 創建功能分支
3. 提交更改
4. 發起合併請求

## 問題回報
如遇到問題，請在 Issues 區提供：
- 問題描述
- 重現步驟
- 期望結果
- 系統環境資訊

## 授權協議
MIT License

## 更新日誌
### 2025/05/01
- 重構專案結構
- 新增 OCR 辨識功能
- 優化預處理流程
- 改進錯誤處理機制
- 更新文件