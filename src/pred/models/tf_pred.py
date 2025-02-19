import numpy as np
import os
import logging
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modeli tek seferlik yükleyelim
MODEL_PATH = "src/pred/models/Trafic_signs_model.h5"
model = None

def load_model_from_path():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        logger.info(f"Loading model from {MODEL_PATH}")
        model = load_model(MODEL_PATH)
    return model

def preprocess_image(image_data):
    try:
        image = Image.open(io.BytesIO(image_data))
        image = image.resize((30, 30))  # Modelin beklediği boyut
        image = np.array(image) / 255.0  # Normalizasyon
        image = np.expand_dims(image, axis=0)  # (1, 30, 30, 3)
        return image
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None

def tf_predict(image):
    try:
        model = load_model_from_path()
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        return {"predicted_class": predicted_class, "confidence": confidence}
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return None
