from fastapi import FastAPI, UploadFile, File, HTTPException
from src.pred.image_classifier import tf_run_classifier

app = FastAPI(title="Traffic Signs Classifier API")

@app.post("/predict/tf/")
async def predict_tf(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    try:
        # Görüntüyü oku
        contents = await file.read()
        
        # Modeli çalıştır ve tahmin al
        prediction = tf_run_classifier(contents)
        
        if prediction is None:
            raise HTTPException(
                status_code=404, detail="Image could not be processed"
            )
            
        return {"prediction": prediction}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
