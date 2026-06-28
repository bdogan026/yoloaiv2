# =============================================================================
# Rapid Response - FTR Docker image (root level - ZORUNLU)
# Hedef: NVIDIA Tesla T4, izole konteyner, calisma aninda internet KAPALI.
# =============================================================================
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Sistem paketleri
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Calisma aninda indirme YAPILMASIN (auto-download kapali, offline garanti).
# Anti-hile NOT: bu env'ler ortam tespiti yapmaz; her kosulda aynidir.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    YOLO_OFFLINE=1 \
    ULTRALYTICS_OFFLINE=1

# Klasor iskeleti
RUN mkdir -p /app/data/input /app/data/output /app/weights /app/src

# Bagimliliklar (build aninda internet var)
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# GOMULU agirliklar: best_model.pt, ocr/, face_landmarker.task
COPY weights/ /app/weights/

# Agirlik yolu celiskisini coz: ornek main.py /app/weights, VM tablosu /app/models.
# /app/models -> /app/weights symlink ile her iki yol da gecerli olur.
RUN ln -sfn /app/weights /app/models

# Seciici COPY (training/ ve live/ image'a GIRMEZ; <8GB icin COPY . . YOK)
COPY config/ /app/config/
COPY src/ /app/src/
COPY main.py .
COPY README.md .

# docker run ile ekstra adim olmadan otomatik baslar
CMD ["python3", "main.py"]
