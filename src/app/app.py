from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
from pred.image_classifier import tf_run_classifier

app = FastAPI(title="Traffic Signs Classifier API")

class PredictionResponse(BaseModel):
    predicted_class: int
    confidence: float
    class_name: str

@app.post("/predict/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        prediction = tf_run_classifier(contents)
        
        if prediction is None:
            raise HTTPException(status_code=400, detail="Image could not be processed")
            
        return PredictionResponse(
            predicted_class=prediction["predicted_class"],
            confidence=prediction["confidence"],
            class_name=prediction["class_name"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7001)