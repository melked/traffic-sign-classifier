import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import logging
import os

# Logger ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model yolu (çevre değişkeninden alınıyor)
MODEL_PATH = os.getenv('MODEL_PATH', 'pred/models/Trafic_signs_model.h5')
model = None

# Modeli yükleme fonksiyonu
def load_model_from_path():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model dosyası bulunamadı: {MODEL_PATH}")
            raise FileNotFoundError(f"Model dosyası bulunamadı: {MODEL_PATH}")
        logger.info(f"Model {MODEL_PATH} yolundan yükleniyor...")
        model = load_model(MODEL_PATH)
    return model

# Görüntüyü ön işleme fonksiyonu
def preprocess_image(image_file):
    try:
        # Görüntüyü açma
        image = Image.open(io.BytesIO(image_file))
        image = image.convert("RGB")  # Renk kanalını RGB'ye dönüştür
        image = image.resize((30, 30))  # Model için gerekli boyut
        image = np.array(image) / 255.0  # Görüntüyü normalize et
        image = np.expand_dims(image, axis=0)  # Modelin beklediği şekil (batch, height, width, channels)
        return image
    except Exception as e:
        logger.error(f"Görüntü işlenirken hata oluştu: {e}")
        return None

# Tahmin fonksiyonu
def tf_predict(image):
    try:
        model = load_model_from_path()  # Modeli yükle
        prediction = model.predict(image)  # Modeli kullanarak tahmin yap
        predicted_class = np.argmax(prediction[0])  # En yüksek olasılığa sahip sınıfı al
        confidence = float(prediction[0][predicted_class])  # Tahminin güven skoru
        return {"predicted_class": int(predicted_class), "confidence": confidence}
    except Exception as e:
        logger.error(f"Tahmin yapılırken hata oluştu: {e}")
        return None

# Görüntüyü alıp tahmin yapan ana fonksiyon
def tf_run_classifier(image_file):
    image = preprocess_image(image_file)  # Görüntüyü ön işleme
    if image is None:
        return None
    return tf_predict(image)  # Tahmini al ve döndür

