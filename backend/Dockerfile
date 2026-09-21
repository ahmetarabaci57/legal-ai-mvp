# 1. Aşama: Resmi hafif Python imajını kullanıyoruz
FROM python:3.11-slim

# 2. Aşama: Python'un logları anlık basması ve .pyc dosyaları üretmemesi için optimizasyon
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 3. Aşama: Bağımlılık katmanını önbelleğe (cache) almak için önce requirements kopyalanır
COPY requirements.txt .

# Güvenlik ve performans için gereksiz paketler temizlenerek yükleme yapılır
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 4. Aşama: Tüm backend kodunun kopyalanması
COPY . .

# Konteynerın dış dünyaya açacağı port (Azure Container Apps bunu otomatik algılayacak)
EXPOSE 8000

# API'yi canlı ortam (Production) modunda ayağa kaldırıyoruz
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
