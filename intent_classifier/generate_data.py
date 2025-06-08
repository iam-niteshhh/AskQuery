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

def generate_synthetic_queries(n=10000):
    """
    Generate synthetic natural language queries for intent classification.

    Args:
        n (int): Number of queries to generate.

    Returns:
        pd.DataFrame: DataFrame with columns 'query' and 'intent'.
    """
    queries = []
    for _ in range(n):
        # Pick a random intent from the defined intent keywords
        intent = random.choice(list(INTENT_KEYWORDS.keys()))
        # Pick a random keyword associated with the intent
        keyword = random.choice(INTENT_KEYWORDS[intent])
        # Pick a random column from the dataset columns
        column = random.choice(list(COLUMNS.keys()))
        # Get the column's data type or possible categories
        col_type = COLUMNS[column]

        # For aggregate intents, only numeric columns make sense
        if intent in ["mean", "sum", "max", "min"]:
            if col_type != "numeric":
                # Retry picking a column until it's numeric
                while col_type != "numeric":
                    column = random.choice(list(COLUMNS.keys()))
                    col_type = COLUMNS[column]
            # Construct a natural language query asking for the aggregate on the column
            query = f"What is the {keyword} {column}?"

        elif intent == "count":
            # Counting number of clients with some attribute
            query = f"{keyword.capitalize()} of clients with {column}"
            # If the column is categorical, add a specific category value to the query
            if col_type != "numeric":
                val = random.choice(col_type)
                query += f" = {val}"

        elif intent == "filter":
            # Filtering rows with column equal to some value
            if col_type == "numeric":
                # Generate a random numeric value for numeric columns
                val = random.randint(1, 100)
            else:
                # Pick a random category for categorical columns
                val = random.choice(col_type)
            query = f"{keyword} {column} = {val}"

        else:
            # Default fallback query format if intent doesn't match above
            query = f"{keyword} {column}"

        # Append the generated query and its intent label
        queries.append((query, intent))

    # Return as a DataFrame for easy downstream ML processing
    return pd.DataFrame(queries, columns=["query", "intent"])

if __name__ == "__main__":
    # Ensure the 'data' directory exists to save the output CSV
    os.makedirs("../Data", exist_ok=True)

    # Generate 1000 synthetic queries
    df = generate_synthetic_queries(1000)

    # Save the generated dataset to CSV for training/testing intent model
    df.to_csv("../Data/intent_dataset.csv", index=False)

    print("Generated intent queries saved to data/intent_dataset.csv")
