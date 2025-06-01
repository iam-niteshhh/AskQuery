import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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




