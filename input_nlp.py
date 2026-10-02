# nlp_utils.py

import re
import string
import numpy as np

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# DOWNLOADS (run once in notebook)
# ------------------------------
# nltk.download('stopwords')

stop_words = set(stopwords.words("english"))


# =====================================================
# 1) BASIC TEXT CLEANING
# =====================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove numbers
    text = re.sub(r"\d+", "", text)

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # remove stopwords
    tokens = [word for word in text.split() if word not in stop_words]
    text = " ".join(tokens)

    return text


# =====================================================
# 2) TF-IDF PROCESSING (FOR CBR MODEL)
# =====================================================
def load_tfidf_model(tfidf_path):
    """
    Loads a saved TF-IDF vectorizer using joblib.
    """
    import joblib
    return joblib.load(tfidf_path)


def tfidf_vectorize(text, tfidf_vectorizer):
    """
    Apply text cleaning + TF-IDF transform.
    """
    cleaned = clean_text(text)
    vector = tfidf_vectorizer.transform([cleaned])
    return vector


# =====================================================
# 3) TOKENIZATION (FOR RNN MODEL)
# =====================================================
def load_tokenizer(tokenizer_path):
    """
    Loads a saved Keras tokenizer.
    """
    import joblib
    return joblib.load(tokenizer_path)


def rnn_prepare_input(text, tokenizer, max_len):
    """
    Clean → tokenize → pad → ready for RNN prediction.
    """
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    return padded


# =====================================================
# 4) FULL PIPELINE WRAPPER (OPTIONAL)
# =====================================================
def prepare_for_models(text, tfidf_vectorizer, tokenizer, max_len):
    """
    Returns both:
    - tfidf_vector (for CBR)
    - padded sequence (for RNN)
    """
    tfidf_vec = tfidf_vectorize(text, tfidf_vectorizer)
    rnn_seq = rnn_prepare_input(text, tokenizer, max_len)
    return tfidf_vec, rnn_seq
