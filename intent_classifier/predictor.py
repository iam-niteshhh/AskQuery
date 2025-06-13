import os
import joblib
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize


stemmer = PorterStemmer()

def stem_tokens(text):
    tokens = word_tokenize(text)
    return [stemmer.stem(token) for token in tokens]


MODELS_DIR = "../models"
MODEL_PATH = os.path.join(MODELS_DIR, "clf_folded_resampled.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer_folded_resampled.joblib")

clf = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def predict_intent(query: str) -> str:
    query_vec = vectorizer.transform([query])
    pred = clf.predict(query_vec)[0]
    return pred

if __name__ == "__main__":
    print("Intent predictor ready. Type queries to predict intent:")
    while True:
        q = input("Query ('exit' to quit): ")
        if q.lower() == "exit":
            break
        print("Predicted intent:", predict_intent(q))
