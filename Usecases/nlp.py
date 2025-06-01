import nltk
import constants

class NLPUseCase:
    def __init__(self, nlp_service, query):
        self.nlp_service = nlp_service
        self.query = query

    def execute(self):

        intent = self.nlp_service.detect_intents(self.query)

        return intent
