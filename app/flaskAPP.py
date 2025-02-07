from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# Initialize the Flask application
app = Flask(__name__)

# Custom Transformer Classes
class TFIDFEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=10000, n_components=300):
        self.max_features = max_features
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words="english", norm=None, use_idf=True, smooth_idf=True, sublinear_tf=False)
        self.svd = TruncatedSVD(n_components=self.n_components)

    def fit(self, X, y=None):
        tfidf_matrix = self.vectorizer.fit_transform(X)
        self.svd.fit(tfidf_matrix)
        return self

    def transform(self, X):
        tfidf_matrix = self.vectorizer.transform(X)
        return self.svd.transform(tfidf_matrix)

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", disable=["parser"])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        sentiment_analyzer = SentimentIntensityAnalyzer()

        for doc in self.nlp.pipe(X, batch_size=2048):
            pos_counts = {
                "nouns": sum(1 for token in doc if token.pos_ == "NOUN"),
                "verbs": sum(1 for token in doc if token.pos_ == "VERB"),
                "adjs": sum(1 for token in doc if token.pos_ == "ADJ"),
                "advs": sum(1 for token in doc if token.pos_ == "ADV"),
                "stopword_ratio": sum(1 for token in doc if token.is_stop) / (len(doc) + 1e-6),
                "avg_word_length": sum(len(token.text) for token in doc) / (len(doc) + 1e-6),
                "ner_person": sum(1 for ent in doc.ents if ent.label_ == "PERSON"),
                "ner_org": sum(1 for ent in doc.ents if ent.label_ == "ORG"),
                "ner_gpe": sum(1 for ent in doc.ents if ent.label_ == "GPE"),
                "sentiment": sentiment_analyzer.polarity_scores(doc.text)['compound'],
            }
            features.append(list(pos_counts.values()))

        return np.array(features)

# Function to clip negative values to zero
def clip_negatives(X):
    return np.maximum(X, 0)

clipper = FunctionTransformer(clip_negatives, validate=False)

# Load the trained model
def load_model():
    try:
        model_path = 'best_model_MultinomialNB.pkl'
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model could not be loaded.'}), 500

    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Check if 'text' key exists
        if 'text' not in data:
            return jsonify({'error': 'No text provided for prediction.'}), 400

        # Extract the text
        text = data['text']

        # Perform prediction
        prediction = model.predict([text])[0]

        # Map prediction to label
        label = 'Real' if prediction == 0 else 'Fake'

        # Return the prediction result
        return jsonify({'prediction': label})

    except ValueError as ve:
        return jsonify({'error': f'Feature mismatch: {str(ve)}. Ensure the input data matches the model requirements.'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
