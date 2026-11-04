# Argus web app (CPU), deployed on Render (see render.yaml).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EASYOCR_MODULE_PATH=/app/.easyocr \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

# OpenCV runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /app

COPY app/requirements.txt app/requirements.txt
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r app/requirements.txt

# Bake the EasyOCR English models into the image so cold starts don't download them
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)"

COPY app app
RUN chown -R user:user /app
USER user

# Render injects $PORT
EXPOSE 10000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
