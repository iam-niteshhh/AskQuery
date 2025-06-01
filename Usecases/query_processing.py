import nltk
from Services.nlp_services import IntentExecutorServices
import constants

class QueryProcessingUseCase:
    def __init__(self, nlp_service, query, dataframe):
        self.nlp_service = nlp_service
        self.query = query
        self.dataframe = dataframe
        self.intent_executor = IntentExecutorServices(
            dataframe=dataframe,
            query=query,
        )

    def execute(self):

        intent = self.nlp_service.detect_intents(self.query)

        matched_columns = []
        matched_columns = self.nlp_service.match_column(
            user_question = self.query,
            df_columns = self.dataframe.columns,
            threshold = constants.COLUMN_MATCH_THRESHOLD,
        )

        result = self.intent_executor.get_intent(intent)

        return result

