Traffic Signs Classifier API
FastAPI tabanlı trafik işareti sınıflandırma API'si. Yüklenen görüntüleri analiz ederek trafik işaretlerini tanımlar ve sınıflandırır.
Kurulum
bashCopy# Virtual environment oluştur (opsiyonel)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Gereksinimleri yükle
pip install -r requirements.txt
Çalıştırma

Model dosyasını pred/models/Trafic_signs_model.h5 konumuna yerleştirin
Uygulamayı başlatın:

bashCopypython main.py

API http://localhost:7001 adresinde çalışmaya başlayacak

Kullanım
bashCopycurl -X POST "http://localhost:7001/predict/" \
     -H "accept: application/json" \
     -F "file=@resim.jpg"
Örnek Yanıt:
jsonCopy{
  "predicted_class": 5,
  "confidence": 0.9916229844093323,
  "class_name": "Speed limit (80km/h)"
}
API dokümantasyonu için: http://localhost:7001/docs
