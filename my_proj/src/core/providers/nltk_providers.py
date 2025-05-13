from dishka import Provider, provide, Scope

from src.domain.nltk_use_cases.nltk import NLTKMethod


class NltkProviders(Provider):
    component = "nltk"
    @provide(scope=Scope.REQUEST)
    def analyze_sentiment_nltk(self) -> NLTKMethod:
        return NLTKMethod()