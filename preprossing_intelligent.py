import pandas as pd
import numpy as np
import re
import ast
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============================================
# LOGGING SETUP
# ============================================
import logging
logging.basicConfig(
    filename='preprocessing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("=" * 80)
logging.info("PREPROCESSING STARTED")
logging.info("=" * 80)

# ============================================
# 1. Download NLTK resources and Setup
# ============================================
try:
    nltk.download('stopwords', quiet=True)
    stop = set(stopwords.words('english'))
    logging.info("✓ NLTK stopwords loaded successfully")
except Exception as e:
    logging.error(f"✗ Failed to load NLTK stopwords: {e}")
    raise

# ============================================
# 2. Define Data Paths
# ============================================
# NOTE: Update these paths to match your local setup
DATA_PATH = "C:/Users/Delta/Documents/datasetML/chatbot/chat/dataset/IT Support Ticket Data.csv"
SAVE_PATH_RNN = "C:/Users/Delta/Documents/datasetML/chatbot/chat/dataforRNN"
SAVE_PATH_CBR = "C:/Users/Delta/Documents/datasetML/chatbot/chat/dataforCBR"

# Create directories if they don't exist
try:
    os.makedirs(SAVE_PATH_RNN, exist_ok=True)
    os.makedirs(SAVE_PATH_CBR, exist_ok=True)
    logging.info(f"✓ Directories created/verified")
except Exception as e:
    logging.error(f"✗ Failed to create directories: {e}")
    raise

print(f"Loading CSV from: {DATA_PATH}")

# ============================================
# 3. Load and Inspect Data
# ============================================
try:
    df = pd.read_csv(DATA_PATH)
    logging.info(f"✓ CSV loaded successfully. Shape: {df.shape}")
    print(f"✓ Data loaded. Shape: {df.shape}")
except FileNotFoundError:
    logging.error(f"✗ CSV file not found at {DATA_PATH}")
    print(f"ERROR: CSV file not found at {DATA_PATH}")
    exit(1)
except Exception as e:
    logging.error(f"✗ Failed to load CSV: {e}")
    print(f"ERROR: Failed to load CSV: {e}")
    exit(1)

# Drop useless columns if they exist
try:
    df = df.drop(columns=["Unnamed: 0"], errors='ignore') 
    logging.info("✓ Dropped unnecessary columns")
except Exception as e:
    logging.error(f"✗ Failed to drop columns: {e}")

# Function to safely convert Tags column from string to list
def convert_tags(tag_string):
    """Safely converts string representation of a list into an actual list."""
    if pd.isna(tag_string) or tag_string == '[]' or not tag_string:
        return []
    if isinstance(tag_string, list):
        return tag_string
    try:
        # Use ast.literal_eval to safely parse stringified lists
        return ast.literal_eval(tag_string)
    except:
        logging.warning(f"Could not convert tag string: {tag_string}")
        return []

try:
    df["Tags"] = df["Tags"].apply(convert_tags)
    logging.info("✓ Tags converted to list format and errors handled")
except Exception as e:
    logging.error(f"✗ Failed to process Tags column: {e}")
    print(f"ERROR: Failed to process Tags column: {e}")
    exit(1)


# Ensure key columns are clean and handle missing values
try:
    df["Body"] = df["Body"].astype(str).fillna("")
    df["Department"] = df["Department"].astype(str).fillna("Unknown")
    df["Priority"] = df["Priority"].astype(str).fillna("Unknown")
    logging.info(f"✓ Key columns processed. Missing values handled.")
except Exception as e:
    logging.error(f"✗ Failed to process key columns: {e}")
    raise

# ============================================
# 4. Data Cleaning Function
# ============================================
def clean_text(text):
    """
    Clean text by converting to lowercase, removing special chars, and stopwords.
    """
    try:
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        words = [w for w in text.split() if w not in stop]
        return " ".join(words)
    except Exception as e:
        logging.warning(f"Error cleaning text: {e}")
        return ""

print("Cleaning text data...")
logging.info("Starting text cleaning...")
try:
    df["Clean_Body"] = df["Body"].apply(clean_text)
    logging.info("✓ Text cleaning completed")
except Exception as e:
    logging.error(f"✗ Failed to clean text: {e}")
    raise

# ============================================
# 5. Data Validation & Dropping Empty Rows
# ============================================
# Drop rows where cleaning resulted in an empty string
initial_shape = df.shape[0]
df = df[df["Clean_Body"].str.len() > 0]
dropped_rows = initial_shape - df.shape[0]

if dropped_rows > 0:
    logging.warning(f"⚠️ Dropped {dropped_rows} empty rows after cleaning.")
    
required_cols = ["Body", "Department", "Priority", "Tags"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    logging.error(f"✗ CRITICAL: Missing required columns: {missing_cols}")
    exit(1)

logging.info(f"✓ All required columns present. Final shape: {df.shape}")

# ============================================
# 6. Encode Labels (Departments) - The Target
# ============================================
try:
    label_enc = LabelEncoder()
    # Use Department as the prediction target for RNN
    df["Dept_enc"] = label_enc.fit_transform(df["Department"])
    
    class_mapping = dict(zip(label_enc.classes_, label_enc.transform(label_enc.classes_)))
    logging.info(f"✓ Labels encoded. Classes: {len(label_enc.classes_)}")
except Exception as e:
    logging.error(f"✗ Failed to encode labels: {e}")
    exit(1)

# ============================================
# SECTION A: PREPARE DATA FOR CBR (TF-IDF)
# ============================================
logging.info("\n--- SECTION A: CBR PREPARATION ---")

try:
    # Step 1: Create the Vectorizer (Using uni/bi-grams for robustness)
    tfidf_cbr = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    # Step 2: Create the vectors
    case_base_vectors = tfidf_cbr.fit_transform(df["Clean_Body"])
    
    # Step 3: Store department, priorities, and TAGS (Solution Hints)
    case_base_solutions = df["Department"].values
    case_base_priorities = df["Priority"].values
    case_base_tags = df["Tags"].values
    
    logging.info(f"✓ CBR Case Base Shape: {case_base_vectors.shape}")
except Exception as e:
    logging.error(f"✗ Failed to prepare CBR data: {e}")
    exit(1)

# ============================================
# SECTION B: PREPARE DATA FOR RNN (Splitting & Tokenizing)
# ============================================
try:
    # Split 1: 80% Train+Val, 20% Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        df["Clean_Body"].values,
        df["Dept_enc"].values,
        test_size=0.2,
        random_state=42,
        stratify=df["Dept_enc"].values
    )
    
    # Split 2: 75% Train, 25% Validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=42,
        stratify=y_temp
    )
    
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    
    train_sequences = tokenizer.texts_to_sequences(X_train)
    val_sequences = tokenizer.texts_to_sequences(X_val)
    test_sequences = tokenizer.texts_to_sequences(X_test)
    
    lengths = [len(s) for s in train_sequences]
    max_length = int(np.percentile(lengths, 95)) + 10
    
    X_train_pad = pad_sequences(train_sequences, maxlen=max_length, padding="post")
    X_val_pad = pad_sequences(val_sequences, maxlen=max_length, padding="post")
    X_test_pad = pad_sequences(test_sequences, maxlen=max_length, padding="post")
    
    logging.info(f"✓ RNN data prepared. Max length: {max_length}. Train shape: {X_train_pad.shape}")
    
except Exception as e:
    logging.error(f"✗ Failed to prepare RNN data: {e}")
    exit(1)


# ============================================
# SECTION C: SAVE ARTIFACTS
# ============================================
logging.info("\n--- SECTION C: SAVING ARTIFACTS ---")

# --- Save RNN Files ---
try:
    # Save Tokenizer, Label Encoder, Numpy Arrays, Metadata
    with open(os.path.join(SAVE_PATH_RNN, 'tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(SAVE_PATH_RNN, 'label_encoder.pickle'), 'wb') as handle:
        pickle.dump(label_enc, handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(os.path.join(SAVE_PATH_RNN, 'X_train_pad.npy'), X_train_pad)
    np.save(os.path.join(SAVE_PATH_RNN, 'y_train.npy'), y_train)
    np.save(os.path.join(SAVE_PATH_RNN, 'X_val_pad.npy'), X_val_pad)
    np.save(os.path.join(SAVE_PATH_RNN, 'y_val.npy'), y_val)
    np.save(os.path.join(SAVE_PATH_RNN, 'X_test_pad.npy'), X_test_pad)
    np.save(os.path.join(SAVE_PATH_RNN, 'y_test.npy'), y_test)
    
    metadata = {
        'max_length': max_length,
        'vocab_size': len(tokenizer.word_index) + 1,
        'num_classes': len(label_enc.classes_),
        'class_names': list(label_enc.classes_)
    }
    with open(os.path.join(SAVE_PATH_RNN, 'metadata.pickle'), 'wb') as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logging.info(f"✓ RNN data saved successfully")
    
except Exception as e:
    logging.error(f"✗ Failed to save RNN files: {e}")
    exit(1)

# --- Save CBR Files ---
try:
    # 4. TF-IDF Vectorizer
    with open(os.path.join(SAVE_PATH_CBR, 'tfidf_vectorizer.pickle'), 'wb') as handle:
        pickle.dump(tfidf_cbr, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info("✓ TF-IDF vectorizer saved")
    
    # 5. Case Base (The Memory) - CRUCIAL: Saving Tags
    with open(os.path.join(SAVE_PATH_CBR, 'case_base_data.pickle'), 'wb') as handle:
        pickle.dump({
            'vectors': case_base_vectors,
            'body_text': df["Body"].values,
            'solutions': case_base_solutions, # Department names are stored here
            'priority': case_base_priorities,
            'tags': case_base_tags             # ✓ Tags saved for Solution Hints
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info("✓ Case base data saved with Tags")
    
except Exception as e:
    logging.error(f"✗ Failed to save CBR files: {e}")
    exit(1)

logging.info("=" * 80)
logging.info("✓ PREPROCESSING COMPLETED SUCCESSFULLY")
logging.info("=" * 80)