import nltk
from nltk import word_tokenize
from nltk.corpus.reader import opinion_lexicon
from nltk.corpus.reader import WordListCorpusReader
from nltk.decorators import new_wrapper

nltk.download('punkt')
nltk.download('opinion_lexicon')


class NLTKMethod:
    def __init__(self, text: str):
        self._text = text
        self._op_lex = opinion_lexicon.OpinionLexiconCorpusReader(root="", fileids=None)



    def analyze_sentiment(self):
        lower_text = self._text.lower()
        words = word_tokenize(text=lower_text, language="russian")

        self._op_lex.words(fileids=words)
        positive_words = self._op_lex.positive()
        negative_words = self._op_lex.negative()


        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        if positive_count > negative_count:
            sentiment = "Положительный"
        elif negative_count > positive_count:
            sentiment = "Отрицательный"
        else:
            sentiment = "Нейтральный"
        print(f"Отзыв - {sentiment}")
        return sentiment

text = input("Введите отзыв: ")
nltk_me = NLTKMethod(text)
try:
    print(nltk_me.analyze_sentiment())
except ValueError as e:
    raise ValueError()
