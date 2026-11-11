import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.getenv("MODEL_DIR", "saved_model")
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at application startup, not on every request.
    state["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_DIR)
    state["model"] = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    state["model"].eval()
    yield
    state.clear()

app = FastAPI(title="Customer Feedback Classifier", lifespan=lifespan)

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Non-empty customer feedback text")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}

@app.post("/predict")
def predict(request: PredictRequest):
    tokenizer = state["tokenizer"]
    model = state["model"]
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
    index = int(torch.argmax(probabilities).item())
    labels = model.config.id2label
    label = labels.get(index, labels.get(str(index), str(index)))
    return {"label": label, "confidence": float(probabilities[index].item())}
