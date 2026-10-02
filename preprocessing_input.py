import re
import nltk
import logging
from nltk.corpus import stopwords

# ============================================
# LOGGING SETUP
# ============================================
logging.basicConfig(
    filename='input_preprocessing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Input preprocessing module initialized")

# ============================================
# NLTK Setup (run once)
# ============================================
try:
    nltk.download('stopwords', quiet=True)
    STOP_WORDS = set(stopwords.words("english"))
    logging.info("✓ NLTK stopwords loaded")
except Exception as e:
    logging.warning(f"Failed to load NLTK stopwords: {e}")
    STOP_WORDS = set()
    print(f"⚠️ Warning: Could not load stopwords: {e}")

# ============================================
# Input Validation Function
# ============================================
def validate_input(user_text, min_length=5, max_length=500):
    """
    Validate user input before processing
    
    Args:
        user_text (str): Raw user input
        min_length (int): Minimum acceptable length
        max_length (int): Maximum acceptable length
    
    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    try:
        # Check if text is provided
        if user_text is None:
            return False, "Input cannot be None"
        
        # Convert to string and strip whitespace
        text = str(user_text).strip()
        
        # Check if empty
        if not text:
            return False, "Input cannot be empty"
        
        # Check minimum length
        if len(text) < min_length:
            return False, f"Input too short (minimum {min_length} characters)"
        
        # Check maximum length
        if len(text) > max_length:
            return False, f"Input too long (maximum {max_length} characters)"
        
        # Check if text contains only whitespace or special characters
        if not re.search(r'[a-zA-Z0-9]', text):
            return False, "Input must contain at least one alphanumeric character"
        
        logging.info(f"✓ Input validation passed: {len(text)} characters")
        return True, None
    
    except Exception as e:
        logging.error(f"Error during input validation: {e}")
        return False, f"Validation error: {str(e)}"

# ============================================
# Text Cleaning Function
# ============================================
def clean_text(user_text):
    """
    Cleans user input text to match preprocessing of training data:
    - Convert to lowercase
    - Remove special characters (keep alphanumeric and spaces)
    - Remove stopwords
    
    Args:
        user_text (str): Raw user input text
    
    Returns:
        str: Cleaned text
    
    Raises:
        ValueError: If input is invalid
    """
    try:
        # Validate input first
        is_valid, error_msg = validate_input(user_text)
        if not is_valid:
            logging.warning(f"Input validation failed: {error_msg}")
            raise ValueError(error_msg)
        
        # Convert to lowercase
        text = str(user_text).lower()
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        # Remove stopwords
        words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]
        
        cleaned = " ".join(words)
        
        if not cleaned:
            logging.warning("Text became empty after cleaning")
            raise ValueError("Text resulted in empty string after cleaning")
        
        logging.info(f"✓ Text cleaned successfully: {len(cleaned)} characters")
        return cleaned
    
    except ValueError as e:
        logging.error(f"Validation error: {e}")
        raise
    except Exception as e:
        logging.error(f"Error cleaning text: {e}")
        raise ValueError(f"Error cleaning text: {str(e)}")

# ============================================
# Text Preprocessing Pipeline
# ============================================
def preprocess_text(user_text, verbose=False):
    """
    Complete preprocessing pipeline for user input
    
    Args:
        user_text (str): Raw user input
        verbose (bool): Print preprocessing steps
    
    Returns:
        dict: Preprocessing results including cleaned text and metrics
    """
    try:
        result = {
            'original': user_text,
            'cleaned': None,
            'is_valid': False,
            'error': None,
            'metrics': {
                'original_length': 0,
                'cleaned_length': 0,
                'words_removed': 0
            }
        }
        
        # Step 1: Validate
        if verbose:
            print("Step 1: Validating input...")
        
        is_valid, error_msg = validate_input(user_text)
        if not is_valid:
            result['error'] = error_msg
            if verbose:
                print(f"  ✗ Validation failed: {error_msg}")
            logging.warning(f"Preprocessing failed: {error_msg}")
            return result
        
        if verbose:
            print("  ✓ Input valid")
        
        result['metrics']['original_length'] = len(str(user_text))
        
        # Step 2: Clean
        if verbose:
            print("Step 2: Cleaning text...")
        
        cleaned_text = clean_text(user_text)
        result['cleaned'] = cleaned_text
        result['is_valid'] = True
        
        if verbose:
            print(f"  ✓ Text cleaned")
        
        # Calculate metrics
        original_words = len(user_text.split())
        cleaned_words = len(cleaned_text.split())
        
        result['metrics']['cleaned_length'] = len(cleaned_text)
        result['metrics']['words_removed'] = original_words - cleaned_words
        
        if verbose:
            print(f"\nPreprocessing Summary:")
            print(f"  Original: {result['metrics']['original_length']} chars, {original_words} words")
            print(f"  Cleaned: {result['metrics']['cleaned_length']} chars, {cleaned_words} words")
            print(f"  Removed: {result['metrics']['words_removed']} words")
        
        return result
    
    except Exception as e:
        logging.error(f"Error in preprocessing pipeline: {e}")
        return {
            'original': user_text,
            'cleaned': None,
            'is_valid': False,
            'error': str(e),
            'metrics': {}
        }

# ============================================
# Text Statistics Function
# ============================================
def get_text_stats(text):
    """
    Get statistics about the text
    
    Args:
        text (str): Text to analyze
    
    Returns:
        dict: Text statistics
    """
    try:
        words = text.split()
        chars = len(text)
        words_count = len(words)
        avg_word_length = chars / words_count if words_count > 0 else 0
        
        stats = {
            'characters': chars,
            'words': words_count,
            'average_word_length': round(avg_word_length, 2),
            'unique_words': len(set(words))
        }
        
        return stats
    except Exception as e:
        logging.error(f"Error getting text stats: {e}")
        return {}

# ============================================
# Example Usage
# ============================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("INPUT PREPROCESSING DEMO")
    print("=" * 80)
    
    # Test cases
    test_inputs = [
        "My laptop won't turn on after the Windows update!",
        "Cannot access VPN from home network",
        "Need help with Excel formulas",
        "x",  # Too short
        "a" * 600,  # Too long
        "!!!",  # No alphanumeric
    ]
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n{'='*80}")
        print(f"Test Case {i}:")
        print(f"Input: {test_input[:100]}...")
        print("-" * 80)
        
        try:
            # Validate
            is_valid, error = validate_input(test_input)
            print(f"Validation: {'✓ PASS' if is_valid else '✗ FAIL'}")
            if error:
                print(f"  Error: {error}")
            
            if is_valid:
                # Preprocess
                result = preprocess_text(test_input, verbose=False)
                
                if result['is_valid']:
                    print(f"\nCleaned Text: {result['cleaned']}")
                    
                    # Get stats
                    stats = get_text_stats(result['cleaned'])
                    print(f"\nStats:")
                    print(f"  Characters: {stats.get('characters', 0)}")
                    print(f"  Words: {stats.get('words', 0)}")
                    print(f"  Unique Words: {stats.get('unique_words', 0)}")
                    print(f"  Avg Word Length: {stats.get('average_word_length', 0)}")
                else:
                    print(f"✗ Preprocessing failed: {result['error']}")
        
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("\n" + "=" * 80)
    print("✓ Input Preprocessing Demo Completed")
    print("=" * 80)