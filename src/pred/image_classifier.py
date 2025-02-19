from PIL import Image
import numpy as np
from .models.tf_pred import tf_predict
import io

def load_image(image_data):
    try:
        image = Image.open(io.BytesIO(image_data))
        image = image.resize((32, 32))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception:
        return None

def tf_run_classifier(image_data):
    img = load_image(image_data)
    if img is None:
        return None
    
    pred_results = tf_predict(img)
    pred_results["status_code"] = 200
    return pred_results