import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
from streamlit import dataframe
import re

import constants

class NLPServices:
    def __init__(self, intent_keywords):
        self.intent_keywords = intent_keywords
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, query):
        tokens = word_tokenize(query.lower())
        return [word for word in tokens if word.isalpha() and word not in self.stop_words]

    def detect_intents(self, query):
        tokens = self.preprocess(query)
        scores = {intent: 0 for intent in self.intent_keywords}

        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if any(keyword in token for token in tokens) or keyword in query.lower():
                    scores[intent] += 1

        best_intent = max(scores, key=scores.get)

        return best_intent if scores[best_intent] > 0 else None

    def match_column(self, user_question, df_columns, threshold=70):
        """
        Match terms from the user question to DataFrame columns using fuzzy string matching.

        Args:
            user_question (str): Natural language input from the user.
            df_columns (list): The actual column names in the DataFrame.
            threshold (int): Minimum fuzzy matching score to consider a column a match.

        Returns:
            list: A list of column names from the DataFrame that best match the question.
        """
        matched_columns = []

        for col in df_columns:
            # Calculate similarity between the column name and the entire user question
            # partial_ratio is used because it can match substrings (e.g., "sales" in "total sales figures")
            score = fuzz.partial_ratio(col.lower(), user_question.lower())

            # Only consider columns with a match score above the threshold
            if score >= threshold:
                matched_columns.append((col, score))

        # Sort matched columns by score in descending order to prioritize the best matches
        matched_columns.sort(key=lambda x: x[1], reverse=True)

        # Return only the column names (not the scores)
        return [col for col, _ in matched_columns]

class IntentExecutorServices:
    def __init__(self, dataframe, query):
        self.intent_handlers_mapping = constants.INTENT_HANDLERS
        self.dataframe = dataframe
        self.query = query.lower()

    def get_intent(self, intent):

        if intent not in self.intent_handlers_mapping:
            return f"Unrecognized intent: {intent}"

        handler_name = self.intent_handlers_mapping[intent]

        # Use Python's dynamic getattr to get the actual method from this class
        handler = getattr(self, handler_name, None)

        if not handler:
            return f"No handler found for intent: {intent}"

        try:
            return handler()

        except Exception as e:
            return f"An error {e} occurred while handling intent '{intent}'"

    def handle_mean_balance(self):
        mean_val = self.dataframe['balance'].mean()
        return f"Average balance is {mean_val:.2f}"

    def handle_count_default(self):
        count = self.dataframe[self.dataframe['default'] == 'yes'].shape[0]
        return f"Number of clients who defaulted: {count}"

    def handle_filter_clients(self):
        # Example: parse filter condition like "joined after 2020"
        year = self.parse_year_filter()
        if year:
            filtered_df = self.dataframe[self.dataframe['join_year'] > year]
            return self.format_dataframe(filtered_df)
        else:
            return "No valid filter condition found."

    def handle_count_subscribed(self):
        count = self.dataframe[self.dataframe['subscribed'] == 'yes'].shape[0]
        return f"Number of clients subscribed: {count}"

    def handle_filter_job_marital(self):
        # Filter for job or marital status keywords in query
        filters = []
        if 'unemployed' in self.query:
            filters.append(self.dataframe['job'] == 'unemployed')
        if 'married' in self.query:
            filters.append(self.dataframe['marital'] == 'married')
        if 'single' in self.query:
            filters.append(self.dataframe['marital'] == 'single')

        if filters:
            combined_filter = filters[0]
            for f in filters[1:]:
                combined_filter |= f
            filtered_df = self.dataframe[combined_filter]
            return self.format_dataframe(filtered_df)
        else:
            return "No job/marital filter found in query."

    # Helper method to parse a year from query for filtering
    def parse_year_filter(self):
        # Simple regex to find "after YYYY" pattern
        match = re.search(r'after (\d{4})', self.query)
        if match:
            return int(match.group(1))
        return None

    # Helper method to format DataFrame nicely
    def format_dataframe(self, df):
        if df.empty:
            return "No results found."
        # Convert to string table, show first 10 rows max
        return df.head(10).to_string(index=False)