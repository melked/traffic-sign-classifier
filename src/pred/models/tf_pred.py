from tensorflow.keras.models import load_model
import numpy as np

def load_model_from_path():
    return load_model("/src/pred/models/Trafic_signs_model.h5")

def tf_predict(image):
    model = load_model_from_path()
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_class])
    return {
        "predicted_class": int(predicted_class),
        "confidence": confidence
    }