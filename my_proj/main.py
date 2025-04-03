from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from textblob import TextBlob

app = FastAPI(
    title="Sentiment Analysis API",
    description="API для анализа тональности текста",
    version="1.0"
)

# Разрешаем CORS (для запросов из браузера)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://127.0.0.1:3000"],  # В продакшене укажите конкретный домен фронтенда!
    allow_methods=["POST"],
    allow_headers=["application/json; charset=utf-8"],
)


class TextRequest(BaseModel):
    text: str


@app.post("/analyze")
async def analyze_sentiment(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")

    analysis = TextBlob(request.text)
    polarity = analysis.sentiment.polarity

    if polarity > 0.1:
        sentiment = "Позитивный"
        color = "#28a745"
    elif polarity < -0.1:
        sentiment = "Негативный"
        color = "#dc3545"
    else:
        sentiment = "Нейтральный"
        color = "#6c757d"

    return {
        "sentiment": sentiment,
        "polarity": round(polarity, 2),
        "color": color
    }
