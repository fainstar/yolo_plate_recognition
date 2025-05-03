FROM python:3.10-slim

# 設定環境變數
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Taipei \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 創建非 root 用戶
RUN useradd -m -s /bin/bash appuser

# 設定工作目錄
WORKDIR /app

# 複製依賴文件
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip3 install --no-cache-dir -r requirements.txt \
    && rm -rf ~/.cache/pip

# 創建模型目錄
RUN mkdir -p /app/models

# 複製模型文件（確保模型文件被複製）
COPY models/best.pt /app/models/

# 複製其他專案文件
COPY . .

# 創建必要的目錄並設定權限
RUN mkdir -p /app/Data /app/OutPut \
    && chown -R appuser:appuser /app

# 設定時區
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 切換到非 root 用戶
USER appuser

# 開放端口
EXPOSE 5000

# 健康檢查
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:5000/ || exit 1

# 啟動應用
CMD ["python3", "main.py"]