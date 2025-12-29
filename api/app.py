from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import predict_sentiment

app = FastAPI(title="Sentiment Analysis API")

class InputText(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running!"}

@app.post("/predict")
def predict(input_data: InputText):
    result = predict_sentiment(input_data.text)
    return result