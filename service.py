import bentoml
import logging
import os
import pandsas as pd
import numpy as np
from bentoml.io import JSON, Text
import requests


logger.getlogger("multi_model_backend")
logger.setLevel(logging.INFO)

text_classifier_bento_model = bentoml.sklearn.get("text_classifier_sklearn:latest")
text_classifier_runner = text_classifier_bento_model.to_runner()
sklearn_vectorizer = text_classifier_bento_model.custom_objects["vectorizer"]

hf_sentiment_model = bentoml.transformers.get("sentiment_analyzer_hf:latest")
hf_sentiment_runner = hf_sentiment_model.to_runner()
hf_tokenizer = hf_sentiment_model.custom_objects["tokenizer"]

spacy_model = spacy.load("en_core_web_sm")

svc = bentoml.Service("multi_model_backend", runners=[text_classifier_runner, hf_sentiment_runner])


@svc.api(input=Text(), output=JSON())
async def classify_text(input_text: str) -> dict:
    logger.info("Classifying text using scikit-learn model...")

    # Preprocess input text
    text_vectorized = sklearn_vectorizer.transform([input_text])

    prediction = await text_classifier_runner.predict.async_run(text_vectorized)

    class_label = "positive" if prediction[0] == 1 else "negative"
    logger.info(f"Scikit-learn model prediction: {class_label}")

    return {"text": input_text, "classification": class_label}


@svc.api(input=Text(), output=JSON())
async def analyze_sentiment(input_text: str) -> dict:
    logger.info("Analyzing sentiment using HF model...")

    inputs = hf_tokenizer(input_text, return_tensors="pt")
    logits = await hf_sentiment_runner.predict.async_run(**inputs)

    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    sentiment_score = float(probabilities[0][1]) # Assuming index 1 is positive sentiment
    sentiment_label = "positive" if sentiment_score > 0.5 else "negative"

    logger.info(f"Sentiment analysis result: {sentiment_label} (score: {sentiment_score:.4f})")
    return {"text": input_text, "sentiment": sentiment_label, "score": sentiment_score}


@svc.on_startup
def load_spacy_model():
    """Load spaCy model once when the service starts."""
    global _spacy_nlp_model
    try:
        _spacy_nlp_model = spacy.load("en_core_web_sm")
        logger.info("SpaCy 'en_core_web_sm' model loaded successfully.")
    except OSError:
        logger.error("SpaCy 'en_core_web_sm' model not found. Attempting download...")
        spacy.cli.download("en_core_web_sm")
        _spacy_nlp_model = spacy.load("en_core_web_sm")
        logger.info("SpaCy 'en_core_web_sm' model downloaded and loaded.")

@svc.api(input=Text(), output=JSON())
async def process_text_spacy(input_text: str) -> dict:
    logger.info(f"Received request for spaCy text processing: '{input_text}'")
    if _spacy_nlp_model is None:
        bentoml.exceptions.ServiceUnavailable("SpaCy model not loaded. Service initializing.")
        return {"error": "SpaCy model not ready."}

    doc = _spacy_nlp_model(input_text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    tokens = [{"text": token.text, "pos": token.pos_, "dep": token.dep_} for token in doc]
    logger.info(f"SpaCy processing completed for: '{input_text}'")
    return {
        "text": input_text,
        "entities": entities,
        "tokens": tokens,
        "sentence_count": len(list(doc.sents))
    }