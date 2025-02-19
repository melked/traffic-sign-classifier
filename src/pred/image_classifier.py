# pred/image_classifier.py
import numpy as np
import os
import logging
from tensorflow.keras.models import load_model
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modeli tek seferlik yükleyelim
MODEL_PATH = os.getenv('MODEL_PATH', 'pred/models/Trafic_signs_model.h5')
model = None

# Trafik işaretlerinin isimlerini içeren sözlük
class_labels = {
    0: 'Speed limit (20km/h)', 
    1: 'Speed limit (30km/h)', 
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 
    4: 'Speed limit (70km/h)', 
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 
    7: 'Speed limit (100km/h)', 
    8: 'Speed limit (120km/h)',
    9: 'No passing', 
    10: 'No passing for vehicles over 3.5 metric tons', 
    11: 'Right-of-way at the next intersection',
    12: 'Priority road', 
    13: 'Yield', 
    14: 'Stop', 
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited', 
    17: 'No entry', 
    18: 'General caution',
    19: 'Dangerous curve to the left', 
    20: 'Dangerous curve to the right', 
    21: 'Double curve',
    22: 'Bumpy road', 
    23: 'Slippery road', 
    24: 'Road narrows on the right',
    25: 'Road work', 
    26: 'Traffic signals', 
    27: 'Pedestrians', 
    28: 'Children crossing',
    29: 'Bicycles crossing', 
    30: 'Beware of ice/snow', 
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits', 
    33: 'Turn right ahead', 
    34: 'Turn left ahead',
    35: 'Ahead only', 
    36: 'Go straight or right', 
    37: 'Go straight or left',
    38: 'Keep right', 
    39: 'Keep left', 
    40: 'Roundabout mandatory',
    41: 'End of no passing', 
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

def load_model_from_path():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {MODEL_PATH}")
        logger.info(f"Model {MODEL_PATH} yolundan yükleniyor...")
        model = load_model(MODEL_PATH)
    return model

def preprocess_image(image_file):
    try:
        image = Image.open(io.BytesIO(image_file))
        image = image.convert("RGB")
        image = image.resize((30, 30))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        logger.error(f"Görüntü işlenirken hata oluştu: {e}")
        return None

def tf_predict(image):
    try:
        model = load_model_from_path()
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        class_name = class_labels.get(predicted_class, "Unknown")
        return {
            "predicted_class": int(predicted_class),
            "confidence": confidence,
            "class_name": class_name
        }
    except Exception as e:
        logger.error(f"Tahmin yapılırken hata oluştu: {e}")
        return None

def tf_run_classifier(image_file):
    image = preprocess_image(image_file)
    if image is None:
        return None
    return tf_predict(image)