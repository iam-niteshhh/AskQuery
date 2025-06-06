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

    def parse_query(self, query, df_columns, threshold = 70):
        tokens = self.preprocess(query)
        action_scores = {intent: 0 for intent in self.intent_keywords}

        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if any(keyword in token for token in tokens) or keyword in query.lower():
                    action_scores[intent] += 1

        best_intent = max(action_scores, key=action_scores.get)
        if action_scores[best_intent] == 0:
            best_intent = None

        matched_column = None
        if best_intent is not None:
            matched_column = self.match_column(query, df_columns, threshold)

        return {
            "action": best_intent,
            "column": matched_column[0] if matched_column else None,
            "column_match_score": matched_column[1] if matched_column else None,
        }

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

        # Return only the column names
        # for now only return the max matched column
        return matched_columns[0]

class IntentExecutorServices:
    def __init__(self, dataframe, query_intent):
        self.dataframe = dataframe
        self.query_intent = query_intent

    def execute(self):
        action = self.query_intent.get("action")
        column = self.query_intent.get("column")

        print(action, column)
        if not action or not column:
            return "Could not understand the intent or column."

        if column not in self.dataframe.columns:
            return f"Column '{column}' not found."

        # Numeric checks for mean and sum
        if action in ["average", "sum"]:
            if not pd.api.types.is_numeric_dtype(self.dataframe[column]):
                return f"Column '{column}' must be numeric for {action} operation."

        if action == "average":
            return self.handle_average(column)
        elif action == "sum":
            return self.handle_sum(column)
        elif action == "count":
            return self.handle_count(column)
        elif action == "filter":
            return self.handle_filter(column)
        else:
            return f"Action '{action}' is not supported yet"

    def handle_average(self, column):
        mean_val = self.dataframe[column].mean()
        return f"Average {column}: {mean_val:.2f}"

    def handle_sum(self, column):
        sum_val = self.dataframe[column].sum()
        return f"Sum of {column}: {sum_val}"

    def handle_count(self, column):
        count_val = self.dataframe[column].count()
        return f"Count of {column}: {count_val}"

    def handle_filter(self, column):
        unique_vals = self.dataframe[column].unique()
        return f"Unique values in {column}: {list(unique_vals)}"
