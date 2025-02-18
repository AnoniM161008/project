

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import nltk
from razdel import tokenize
from nltk.corpus import opinion_lexicon
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
app = FastAPI()

# Модель для входных данных
class TextInput(BaseModel):
    text: str

# Подключение статических файлов
app.mount("/static", StaticFiles(directory="static"), name="static")

# Маршрут для анализа тональности
@app.post("/analyze/")
async def analyze_sentiment(text_input: TextInput):
    text = text_input.text

    # Здесь ваш код для анализа тональности
    # Например:
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

    # Замените на реальный анализ
    def analyze_sentiment_vader(self, text: str):
        vader_score = self.vader_analyzer.polarity_scores(text)
        if vader_score['compound'] >= 0.05:
            sentiment = "Положительный"
        elif vader_score['compound'] <= -0.05:
            sentiment = "Отрицательный"
        else:
            sentiment = "Нейтральный"

        return sentiment, vader_score

    # Замените на реальный анализ
    vader_score = {"neg": 0.1, "neu": 0.8, "pos": 0.1, "compound": 0.0}  # Замените на реальный анализ

    return {
        "nltk_sentiment": nltk_sentiment,
        "vader_sentiment": vader_sentiment,
        "vader_score": vader_score
    }