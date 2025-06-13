import nltk
from Services.nlp_services import IntentExecutorServices
import constants
from Services.charts_services import VisualizationServices


class QueryProcessingUseCase:
    def __init__(self, nlp_service, query, dataframe,column_classification):
        self.nlp_service = nlp_service
        self.query = query
        self.dataframe = dataframe
        self.intent_executor = None
        self.visualizer = None
        self.column_classification = column_classification

    def execute(self):

        parsed_intent = self.nlp_service.parse_query(
            query=self.query,
            dataframe=self.dataframe,
            df_columns=self.dataframe.columns,
            threshold=constants.COLUMN_MATCH_THRESHOLD,
            column_classification = self.column_classification
        )
        print(parsed_intent)

        self.intent_executor = IntentExecutorServices(
            dataframe=self.dataframe,
            query_intent=parsed_intent,
        )
        result = self.intent_executor.execute()

        self.visualizer = VisualizationServices(
            dataframe=self.dataframe,
            query_intent=parsed_intent,
        )
        fig, file_message = self.visualizer.execute()
        return result, fig, file_message

