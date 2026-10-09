# Воспроизводимое окружение: docker build -t pulsar . && docker run --rm -v $PWD/data:/app/data -v $PWD/outputs:/app/outputs pulsar make all
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends make curl libarchive-tools gdal-bin libgdal-dev build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app/src PY=python
CMD ["make", "test"]
