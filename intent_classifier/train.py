import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Paths for data and saved models
DATA_PATH = "../Data/intent_dataset.csv"
MODELS_DIR = "../models"

def train(use_k_fold = False, n_splits = 0):
    # Load dataset containing text queries and their intent labels
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates(subset=["query", "intent"])
    X, y = df["query"], df["intent"]

    if use_k_fold and n_splits > 0:
        X, y = X.to_numpy(), y.to_numpy()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold = 1
        reports = []
        for train_index, test_index in skf.split(X, y):
            print(f"\n Fold {fold}")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            vectorizer = TfidfVectorizer()
            X_train = vectorizer.fit_transform(X_train)
            X_test = vectorizer.transform(X_test)

            clf = LogisticRegression(random_state=42, max_iter=500)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            report = classification_report(y_test, y_pred, output_dict=True)
            reports.append(report)
            print(classification_report(y_test, y_pred))

            fold += 1

        macro_f1s = [rep["macro avg"]["f1-score"] for rep in reports]
        print(f"\n✅ Average Macro F1-Score over {n_splits} folds: {np.mean(macro_f1s):.4f}")
        model_name = "clf_folded.joblib"
        vector_name = 'vectorizer_folded.joblib'

    else:
        # Split data into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Convert text to TF-IDF features
        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)  # Fit on train and transform
        X_test_vec = vectorizer.transform(X_test)        # Transform test data

        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)
        print(classification_report(y_test, y_pred))

        # random_forest = RandomForestClassifier(random_state=42)
        # random_forest.fit(X_train_vec, y_train)
        # y_pred = random_forest.predict(X_test_vec)
        # print(classification_report(y_test, y_pred))
        overlap = set(X_train).intersection(set(X_test))
        print(f"Overlapping samples between train and test: {len(overlap)}")
        # Save the model and vectorizer for later use
        model_name = "clf.joblib"
        vector_name = 'vectorizer.joblib'

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODELS_DIR, model_name))
    # joblib.dump(random_forest, os.path.join(MODELS_DIR, "random_forest.joblib"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, vector_name))
print(f"Saved model and vectorizer to {MODELS_DIR}")

if __name__ == "__main__":
    train(n_splits=10, use_k_fold=True)
