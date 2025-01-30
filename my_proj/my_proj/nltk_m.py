import nltk
from nltk import word_tokenize
from nltk.corpus.reader import opinion_lexicon

nltk.download('punkt')
nltk.download('opinion_lexicon')


class NLTKMethod:
    def __init__(self):
        self._positive_words = set(opinion_lexicon.positive())
        self._negative_words = set(opinion_lexicon.negative())

    def analyze_sentiment(self, text: str):
        text = text.lower()
        words = word_tokenize(text)

        positive_count = sum(1 for word in words if word in self._positive_words)
        negative_count = sum(1 for word in words if word in self._negative_words)
        if positive_count > negative_count:
            sentiment = "Положительный"
        elif negative_count > positive_count:
            sentiment = "Отрицательный"
        else:
            sentiment = "Нейтральный"
        print(f"Отзыв - {sentiment}")
        return sentiment

nltk_me = NLTKMethod()
try:
    print(nltk_me.analyze_sentiment())
except ValueError as e:
    raise ValueError()
