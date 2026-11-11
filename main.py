from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Transformers API")

# Load models once at startup, not on every request
sentiment = pipeline("sentiment-analysis", model="sst2_finetuned")
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

class TextIn(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/sentiment")
def predict_sentiment(item: TextIn):
    result = sentiment(item.text)[0]
    return {"label": result["label"], "score": float(result["score"])}

@app.post("/ner")
def predict_ner(item: TextIn):
    entities = ner(item.text)
    return {
        "entities": [
            {
                "entity": e["entity_group"],
                "word": e["word"],
                "score": float(e["score"]),
                "start": int(e["start"]),
                "end": int(e["end"]),
            }
            for e in entities
        ]
    }