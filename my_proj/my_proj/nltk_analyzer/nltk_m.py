import nltk
from razdel import tokenize
from nltk.corpus import opinion_lexicon
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rusentiment import RuSentiment

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('opinion_lexicon')
rusentiment = RuSentiment()
class NLTKMethod:
    def __init__(self):
        self._positive_words = set(opinion_lexicon.positive())
        self._negative_words = set(opinion_lexicon.negative())
        self.vader_analyzer = SentimentIntensityAnalyzer()

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

        print(f"NLTK Отзыв - {sentiment}")
        return sentiment

    def analyze_sentiment_vader(self, text: str):
        # Анализ с использованием VADER
        vader_score = self.vader_analyzer.polarity_scores(text)
        if vader_score['compound'] >= 0.05:
            sentiment = "Положительный"
        elif vader_score['compound'] <= -0.05:
            sentiment = "Отрицательный"
        else:
            sentiment = "Нейтральный"

        print(f"VADER Отзыв - {sentiment} (Оценка: {vader_score})")
        return sentiment

    def analyze_sentiment_rusentiment(text: str):
        result = rusentiment.predict(text)
        if result == "positive":
            return "Положительный"
        elif result == "negative":
            return "Отрицательный"
        else:
            return "Нейтральный"

        sentiment = analyze_sentiment_rusentiment(text)
        print(f"Текст: {text}")
        print(f"Тональность: {sentiment}")
        print("-" * 30)

nltk_me = NLTKMethod()
try:
    text = input('Введите текст: ')
    nltk_me.analyze_sentiment_nltk(text)
    nltk_me.analyze_sentiment_vader(text)
except ValueError as e:
    raise ValueError()