from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
from pred.image_classifier import tf_run_classifier

# FastAPI uygulaması başlatılıyor
app = FastAPI(title="Traffic Signs Classifier API")

# Pydantic modelini oluşturuyoruz (yanıt formatı)
class PredictionResponse(BaseModel):
    predicted_class: int
    confidence: float

@app.post("/predict/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        # Yüklenen dosyayı alıyoruz
        contents = await file.read()
        # Görüntü verisini işleyip tahmin alıyoruz
        prediction = tf_run_classifier(contents)

        # Tahmin yapılamazsa hata döndürüyoruz
        if prediction is None:
            raise HTTPException(status_code=400, detail="Image could not be processed")

        # Tahmin sonuçlarını döndürüyoruz
        return PredictionResponse(
            predicted_class=prediction["predicted_class"],
            confidence=prediction["confidence"]
        )
    except Exception as e:
        # Genel hataları yakalayıp HTTPException döndürüyoruz
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Uvicorn ile FastAPI uygulamasını başlatıyoruz
    uvicorn.run(app, host="0.0.0.0", port=7001)
