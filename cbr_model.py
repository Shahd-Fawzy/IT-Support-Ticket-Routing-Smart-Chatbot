import pickle
import numpy as np
import re
import nltk
import logging
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

# ============================================
# LOGGING SETUP
# ============================================
logging.basicConfig(
    filename='cbr_retrieval.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("CBR Retrieval module initialized")

# ============================================
# Load Stored CBR Artifacts
# ============================================
CBR_PATH = "C:/Users/Delta/Documents/datasetML/chatbot/chat/dataforCBR"

try:
    # Load TF-IDF Vectorizer
    with open(f"{CBR_PATH}/tfidf_vectorizer.pickle", "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    logging.info("✓ TF-IDF vectorizer loaded")
    
    # Load Case Base
    with open(f"{CBR_PATH}/case_base_data.pickle", "rb") as f:
        case_base = pickle.load(f)
    
    case_vectors = case_base["vectors"]
    case_texts = case_base["body_text"]
    case_solutions = case_base["solutions"]
    case_priority = case_base["priority"]
    
    logging.info(f"✓ Case base loaded. Size: {case_vectors.shape}")
    print("✓ CBR artifacts loaded successfully")
    
except FileNotFoundError as e:
    logging.error(f"✗ CBR files not found: {e}")
    print(f"ERROR: CBR files not found at {CBR_PATH}")
    print("Make sure preprocessing.py has been run first.")
    exit(1)
except Exception as e:
    logging.error(f"✗ Failed to load CBR artifacts: {e}")
    print(f"ERROR: Failed to load CBR artifacts: {e}")
    exit(1)

# ============================================
# Setup NLTK Stopwords
# ============================================
try:
    nltk.download('stopwords', quiet=True)
    stop = set(stopwords.words('english'))
    logging.info("✓ NLTK stopwords loaded")
except Exception as e:
    logging.warning(f"⚠️ Failed to load stopwords: {e}")
    stop = set()

# ============================================
# Text Cleaning Function
# ============================================
def clean_text(text):
    """
    Clean text using same method as preprocessing:
    - Convert to lowercase
    - Remove special characters
    - Remove stopwords
    """
    try:
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        words = [w for w in text.split() if w not in stop]
        cleaned = " ".join(words)
        return cleaned if cleaned.strip() else ""
    except Exception as e:
        logging.warning(f"Error cleaning text: {e}")
        return ""

# ============================================
# CBR Retrieval Function
# ============================================
def retrieve_similar_cases(query_text, k=3, min_similarity=0.2):
    """
    Retrieve k most similar cases from case base using TF-IDF + Cosine Similarity
    
    Args:
        query_text (str): User's query/problem description
        k (int): Number of similar cases to retrieve (default: 3)
        min_similarity (float): Minimum similarity threshold (0-1)
    
    Returns:
        list: List of dictionaries containing similar cases
    
    Raises:
        ValueError: If query is empty or invalid
    """
    # Input validation
    if not query_text or not isinstance(query_text, str):
        logging.warning("Invalid query input")
        raise ValueError("Query must be a non-empty string")
    
    if k < 1 or k > len(case_vectors):
        logging.warning(f"Invalid k value: {k}")
        k = min(3, len(case_vectors))
    
    if min_similarity < 0 or min_similarity > 1:
        logging.warning(f"Invalid similarity threshold: {min_similarity}")
        min_similarity = 0.2
    
    try:
        # Clean and vectorize query
        clean_query = clean_text(query_text)
        
        if not clean_query:
            logging.warning("Query resulted in empty string after cleaning")
            return []
        
        query_vec = tfidf_vectorizer.transform([clean_query])
        
        # Calculate cosine similarity with all cases
        sims = cosine_similarity(query_vec, case_vectors)[0]
        
        # Get top k indices
        top_idx = sims.argsort()[-k:][::-1]
        
        # Build results
        results = []
        for i in top_idx:
            similarity = float(sims[i])
            
            # Only include if above threshold
            if similarity >= min_similarity:
                results.append({
                    "similarity": similarity,
                    "body": str(case_texts[i])[:500],  # Limit text length
                    "department": str(case_solutions[i]),
                    "priority": str(case_priority[i])
                })
        
        logging.info(f"Retrieved {len(results)} cases for query (threshold: {min_similarity})")
        
        if not results:
            logging.warning(f"No cases found above similarity threshold {min_similarity}")
        
        return results
    
    except Exception as e:
        logging.error(f"Error during case retrieval: {e}")
        print(f"ERROR in CBR retrieval: {e}")
        return []

# ============================================
# Statistics Function
# ============================================
def get_cbr_statistics():
    """
    Get statistics about the case base
    
    Returns:
        dict: Statistics including size, departments, priorities
    """
    try:
        unique_depts = len(set(case_solutions))
        unique_priorities = len(set(case_priority))
        
        stats = {
            "total_cases": case_vectors.shape[0],
            "vector_dimensions": case_vectors.shape[1],
            "unique_departments": unique_depts,
            "unique_priorities": unique_priorities,
            "departments": list(set(case_solutions)),
            "priorities": list(set(case_priority))
        }
        
        logging.info(f"CBR Statistics: {stats}")
        return stats
    except Exception as e:
        logging.error(f"Error getting statistics: {e}")
        return {}

# ============================================
# Example Usage
# ============================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CBR RETRIEVAL SYSTEM")
    print("=" * 80)
    
    # Display statistics
    try:
        stats = get_cbr_statistics()
        print(f"\nCBR Case Base Statistics:")
        print(f"  Total Cases: {stats.get('total_cases', 0)}")
        print(f"  Vector Dimensions: {stats.get('vector_dimensions', 0)}")
        print(f"  Unique Departments: {stats.get('unique_departments', 0)}")
        print(f"  Unique Priorities: {stats.get('unique_priorities', 0)}")
    except Exception as e:
        print(f"Could not display statistics: {e}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE QUERIES")
    print("=" * 80)
    
    # Test queries
    test_queries = [
        "My VPN is not connecting to the company network",
        "Email keeps crashing when sending attachments",
        "Cannot access shared network drive from home"
    ]
    
    for query in test_queries:
        print(f"\n📌 Query: {query}")
        print("-" * 80)
        
        try:
            results = retrieve_similar_cases(query, k=3, min_similarity=0.1)
            
            if results:
                for idx, case in enumerate(results, 1):
                    print(f"\n  Case {idx}:")
                    print(f"    Similarity: {case['similarity']:.3f}")
                    print(f"    Department: {case['department']}")
                    print(f"    Priority: {case['priority']}")
                    print(f"    Text: {case['body'][:150]}...")
            else:
                print("  ⚠️ No similar cases found")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 80)
    print("✓ CBR Retrieval Demo Completed")
    print("=" * 80)