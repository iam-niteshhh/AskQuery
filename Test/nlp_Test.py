import constants
import nltk
from Usecases.query_processing import QueryProcessingUseCase
from Services.nlp_services import NLPServices
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



if __name__ == "__main__":
    nlp_services = NLPServices(
        intent_keywords=constants.INTENT_KEYWORDS,
    )
    for query in constants.QUERIES:
        nlpqueryprocessing_usecase = QueryProcessingUseCase(
            nlp_service=nlp_services,
            query=query
        )
        nlpqueryprocessing_usecase.execute()