import constants
import nltk
from Usecases.nlp import NLPUseCase
from Services.nlp import NLPServices
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



if __name__ == "__main__":
    nlp_services = NLPServices(
        intent_keywords=constants.intent_keywords,
    )
    for query in constants.queries:
        nlp_usecase = NLPUseCase(
            nlp_service=nlp_services,
            query=query
        )
        nlp_usecase.execute()