# ---- Builder Stage ----
FROM python:3.10-slim as builder

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Taipei \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1

# 安裝編譯時和運行時的系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    # build-essential \ # 如果 requirements.txt 中有需要編譯的套件，則取消註解
    # 運行時系統依賴 (根據實際需求調整)
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libfreetype6 \
    curl \
    # 新增: OpenCV 和 PaddleOCR 可能需要的其他依賴
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

COPY requirements.txt .
# 安裝 torch CPU 版本
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# 安裝其他 Python 依賴
RUN pip3 install --no-cache-dir -r requirements.txt \
    && rm -rf ~/.cache/pip

# 複製應用程式碼 (只複製必要的)
COPY main.py .
COPY config /app/config
COPY services /app/services
COPY utils /app/utils
COPY templates /app/templates
COPY models/best.pt /app/models/best.pt

# ---- Final Stage ----
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Taipei \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1

# 安裝運行時必要的系統依賴 (應該是 builder stage 中運行時依賴的子集)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libfreetype6 \
    curl \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN useradd -m -s /bin/bash appuser
WORKDIR /app

# 從 builder stage 複製已安裝的套件和必要的二進制文件
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 從 builder stage 複製應用程式碼和模型
COPY --from=builder /app/main.py .
COPY --from=builder /app/config /app/config
COPY --from=builder /app/services /app/services
COPY --from=builder /app/utils /app/utils
COPY --from=builder /app/templates /app/templates
COPY --from=builder /app/models/best.pt /app/models/best.pt

RUN mkdir -p /app/Data /app/OutPut \
    && chown -R appuser:appuser /app

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

USER appuser
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:5000/ || exit 1
CMD ["python3", "main.py"]