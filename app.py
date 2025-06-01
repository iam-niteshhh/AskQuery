import streamlit as st
import pandas as pd

import constants
from Services.data import DataService
from Usecases.data_handler import DataUsecase
from Usecases.query_processing import QueryProcessingUseCase
from Services.nlp_services import NLPServices
st.set_page_config(page_title="AskQuery", layout="wide")
st.title("AskQuery – Ask Questions About Bank Data")

file_path = "./Data"
zip_file_name = "bank+marketing.zip"
csv_file_name = "bank.csv"

try:
    data_handler_usecase = DataUsecase(
        file_path=file_path,
        zip_file_name=zip_file_name,
        csv_file_name=csv_file_name,
    )
    status, data_set = data_handler_usecase.execute()
    st.success("File loaded successfully!")

    if status:
        if data_set is not None:
            st.write("Here is a preview of your data:")
            st.dataframe(data_set.head(10))

            # Step 2: Enter natural language query
            query = st.text_input("Ask a question about your data:")

            if query:
                st.markdown("#### Your Query:")
                st.write(query)

                nlp_services = NLPServices(intent_keywords=constants.INTENT_KEYWORDS)

                # Step 3: Process the query
                queryprocessing_usecase = QueryProcessingUseCase(
                    nlp_service=nlp_services,
                    query=query,
                    dataframe=data_set,
                )
                result = queryprocessing_usecase.execute()

                st.markdown("#### Result:")
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result)
                else:
                    st.write(result)
        else:
            st.error("Could not read file. Make sure it's a valid zip or CSV.")
except Exception as e:
    st.error("Looks like something went wrong.")
    st.exception(e)