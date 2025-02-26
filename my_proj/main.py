from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from razdel import tokenize
from nltk.corpus import opinion_lexicon
import nltk
import os

# Загрузка необходимых ресурсов NLTK
nltk.download('opinion_lexicon')

# Создание экземпляра FastAPI
app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить все домены (для разработки)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модель для входных данных
class TextInput(BaseModel):
    text: str

# Полный путь к папке statics
STATIC_DIR = os.path.join(os.path.dirname(__file__), "statics")

# Подключение статических файлов
app.mount("/statics", StaticFiles(directory=STATIC_DIR), name="statics")

# Инициализация анализатора VADER
vader_analyzer = SentimentIntensityAnalyzer()

# Инициализация NLTK
class NLTKMethod:
    def __init__(self):
        self._positive_words = set(opinion_lexicon.positive())
        self._negative_words = set(opinion_lexicon.negative())

    def analyze_sentiment_nltk(self, text: str):
        text = text.lower()
        words = [token.text for token in tokenize(text)]

        positive_count = sum(1 for word in words if word in self._positive_words)
        negative_count = sum(1 for word in words if word in self._negative_words)

        if positive_count > negative_count:
            sentiment = "Положительный"
        elif negative_count > positive_count:
            sentiment = "Отрицательный"
        else:
            sentiment = "Нейтральный"

        return sentiment

# Инициализация NLTK анализатора
nltk_me = NLTKMethod()

# Маршрут для анализа тональности
@app.post("/analyze/")
async def analyze_sentiment(text_input: TextInput):
    text = text_input.text

    # Анализ тональности с использованием NLTK
    nltk_sentiment = nltk_me.analyze_sentiment_nltk(text)

    # Анализ тональности с использованием VADER
    vader_score = vader_analyzer.polarity_scores(text)
    if vader_score['compound'] >= 0.05:
        vader_sentiment = "Положительный"
    elif vader_score['compound'] <= -0.05:
        vader_sentiment = "Отрицательный"
    else:
        vader_sentiment = "Нейтральный"

    return {
        "nltk_sentiment": nltk_sentiment,
        "vader_sentiment": vader_sentiment,
        "vader_score": vader_score
    }