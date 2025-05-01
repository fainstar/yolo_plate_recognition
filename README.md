# 車牌辨識系統

## 專案簡介
本專案是一個基於 YOLOv8 模型的車牌辨識系統，提供 Web 介面和 API 服務，支援 GPU 加速（如果可用），並自動回退到 CPU 運算。系統使用 base64 進行圖片傳輸，確保資料傳輸的安全性和效率。

## 專案結構
```
├── main.py                 # 主程式（API 服務）
├── Base/                   # 基礎配置
│   ├── environment.txt     # 環境依賴
│   └── README.md          # 說明文件
├── Data/                   # 資料目錄
│   ├── base_01.png        # 測試圖片
│   └── data03.jpeg        # 測試圖片
├── OutPut/                # 輸出目錄
│   └── result.jpg         # 辨識結果
├── templates/             # 網頁模板
│   └── index.html         # Web 介面
└── Test/                  # 測試目錄
    ├── test_api.py        # API 測試
    └── test_base.py       # 基礎功能測試
```

## 功能特點
- Web 介面支援拖放上傳圖片
- RESTful API 支援 base64 圖片傳輸
- GPU 加速支援（自動回退到 CPU）
- 即時車牌偵測和標記
- 支援批次處理
- 提供完整的測試案例

## 環境需求
- Python 3.10 或以上
- CUDA 11.6（用於 GPU 加速）
- onnxruntime-gpu 1.15.1

## 安裝步驟
1. 安裝依賴套件：
```bash
pip install -r Base/environment.txt
```

## 使用方法

### Web 介面
1. 啟動服務：
```bash
python main.py
```
2. 開啟瀏覽器訪問：`http://localhost:5000`
3. 拖放或點擊上傳圖片
4. 等待處理完成，查看標記結果

### API 使用
- 端點：`http://localhost:5000/infer`
- 方法：POST
- 請求格式：
```json
{
    "image": "base64編碼的圖片"
}
```
- 回應格式：
```json
{
    "success": true,
    "image": "base64編碼的處理結果圖片"
}
```

### API 測試
執行測試腳本：
```bash
python Test/test_api.py
```
測試結果將保存在 OutPut/result.jpg

## 效能優化
- 模型只會載入一次，避免重複載入
- 使用 GPU 加速（如果可用）
- 支援批次處理多張圖片

## 測試
執行所有測試：
```bash
python -m pytest Test/
```

## 注意事項
- 確保 CUDA 驅動程式與 onnxruntime-gpu 版本相容
- 圖片大小建議不超過 4MB
- 支援的圖片格式：JPG、PNG
- 首次執行時模型載入可能需要較長時間
- 建議在有 GPU 的環境下運行以獲得最佳效能

## 更新日誌
- 2025/05/01
  - 更新專案結構
  - 優化圖片處理流程
  - 改進錯誤處理
  - 添加完整的測試案例

## License
MIT License