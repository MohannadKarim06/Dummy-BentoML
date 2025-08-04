import bentoml
import logging 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import spacy

def train_and_save_sklearn_model():

    print("Training and saving scikit-learn text classification model...")
    # Dummy data for demonstration
    texts = ["I love this product", "This is terrible", "It's okay, not great", "Fantastic experience!"]
    labels = [1, 0, 0, 1] # 1 for positive, 0 for negative

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    clf = LogisticRegression()
    clf.fit(X, labels)

    bentoml.sklearn.save_model("text_classifier_sklearn", clf,
                               custom_objects={"vectorizer": vectorizer})
    print("Scikit-learn model saved as 'text_classifier_sklearn'.")


def save_spacy_model():

    print("Saving spacy model...")

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spacy model...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")


def save_hf_sentiment_model():
    print("Saving Hugging Face sentiment analysis model...")
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    bentoml.transformers.save_model("sentiment_analyzer_hf", model,
                                   custom_objects={"tokenizer": tokenizer})
    print(f"Hugging Face model '{model_name}' saved as 'sentiment_analyzer_hf'.")

# --- Run the saving operations ---
if __name__ == "__main__":
    train_and_save_sklearn_model()
    save_spacy_model() 
    save_hf_sentiment_model()
    print("\nAll models prepared and saved to BentoML Model Store (or noted for dynamic loading).")