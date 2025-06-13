import random
import pandas as pd
import os

# Mapping of intent types to keywords commonly used in queries
INTENT_KEYWORDS = {
    "mean": ["average", "mean", "avg"],
    "sum": ["sum", "total", "add up", "aggregate"],
    "count": ["count", "number of", "how many"],
    "max": ["maximum", "max", "highest", "top"],
    "min": ["minimum", "min", "lowest", "bottom"],
    "filter": ["filter", "show", "only", "where", "with"]
}

# Dataset columns and their types or possible categorical values
COLUMNS = {
    "age": "numeric",  # numeric columns will be used with numeric queries
    "job": ["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur",
            "student", "blue-collar", "self-employed", "retired", "technician", "services"],
    "marital": ["married", "divorced", "single"],
    "education": ["unknown", "secondary", "primary", "tertiary"],
    "default": ["yes", "no"],
    "balance": "numeric",
    "housing": ["yes", "no"],
    "loan": ["yes", "no"],
    "contact": ["unknown", "telephone", "cellular"],
    "day": "numeric",
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "duration": "numeric",
    "campaign": "numeric",
    "pdays": "numeric",
    "previous": "numeric",
    "poutcome": ["unknown", "other", "failure", "success"]
}

def classify_columns(df, unique_ratio_thresh=0.05, max_unique_values=20):
    """
    Automatically classify columns in a DataFrame as 'categorical' or 'continuous'.

    Returns:
        dict: column_name -> 'categorical' or 'continuous'
    """
    classification = {}

    for col in df.columns:
        series = df[col]
        nunique = series.nunique()
        total = len(series)
        unique_ratio = nunique / total

        if not pd.api.types.is_numeric_dtype(series):
            classification[col] = 'categorical'
        elif nunique <= max_unique_values or unique_ratio < unique_ratio_thresh:
            classification[col] = 'categorical'
        else:
            classification[col] = 'continuous'

    return classification

def generate_synthetic_queries(data_frame, n=10000, columns=COLUMNS):
    """
    Generate balanced synthetic queries for each intent type.

    Args:
        n (int): Total number of queries to generate.

    Returns:
        pd.DataFrame: DataFrame with columns 'query' and 'intent'.
    """
    queries = []
    intents = list(INTENT_KEYWORDS.keys())
    n_per_intent = n // len(intents)

    for intent in intents:
        count = 0
        while count < n_per_intent:
            keyword = random.choice(INTENT_KEYWORDS[intent])
            column = random.choice(list(columns.keys()))
            col_type = columns[column]

            # Handle intents only valid for numeric columns
            if intent in ["mean", "sum", "max", "min"]:
                if col_type != "continuous":
                    continue
                query = f"What is the {keyword} {column}?"

            elif intent == "count":
                query = f"{keyword.capitalize()} of clients with {column}"
                if col_type == "categorical":
                    val = random.choice(data_frame[column].dropna().unique())
                    query += f" = {val}"

            elif intent == "filter":
                if col_type == "categorical":
                    val = random.choice(data_frame[column].dropna().unique())
                else:
                    val = float(data_frame[column].dropna().sample(1).values[0])
                query = f"{keyword} {column} = {val}"

            else:
                query = f"{keyword} {column}"

            queries.append((query, intent))
            count += 1

    return pd.DataFrame(queries, columns=["query", "intent"])

if __name__ == "__main__":
    # Ensure the 'data' directory exists to save the output CSV
    os.makedirs("../Data", exist_ok=True)
    data_frame = pd.read_csv('../Data/bank-full.csv', delimiter=';')

    categorical_data = classify_columns(data_frame, unique_ratio_thresh=0.05, max_unique_values=20)
    # print(categorical_data)
    # # Generate 1000 synthetic queries
    df = generate_synthetic_queries(n=3000, columns=categorical_data, data_frame = data_frame)
    #
    # # Save the generated dataset to CSV for training/testing intent model
    df.to_csv("../Data/intent_dataset_new.csv", index=False)

    print("Generated intent queries saved to data/intent_dataset.csv")
