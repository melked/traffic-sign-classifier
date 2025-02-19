# Python 3.9 slim tabanlı bir imaj kullanıyoruz
FROM python:3.9-slim

# Çalışma dizinini belirle
WORKDIR /app

# Bağımlılıkları yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kaynak kodlarını konteynere kopyala
COPY ./src /app/src

# Model dosyasını kopyala
COPY ./src/pred/models/Trafic_signs_model.h5 /app/src/pred/models/Trafic_signs_model.h5

# Python yoluna src klasörünü ekleyelim
ENV PYTHONPATH=/app/src

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "7001", "--reload"]
