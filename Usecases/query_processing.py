import nltk
from Services.nlp_services import IntentExecutorServices
import constants

class QueryProcessingUseCase:
    def __init__(self, nlp_service, query, dataframe):
        self.nlp_service = nlp_service
        self.query = query
        self.dataframe = dataframe
        self.intent_executor = None

    def execute(self):

        parsed_intent = self.nlp_service.parse_query(
            query=self.query,
            df_columns=self.dataframe.columns,
            threshold=constants.COLUMN_MATCH_THRESHOLD,
        )
        print(parsed_intent)

        self.intent_executor = IntentExecutorServices(
            dataframe=self.dataframe,
            query_intent=parsed_intent,
        )
        result = self.intent_executor.execute()

        return result

