import pickle
import os
import numpy as np
import logging
from datetime import datetime
from scipy.sparse import vstack

# ============================================
# LOGGING SETUP
# ============================================
logging.basicConfig(
    filename='feedback.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Feedback module initialized")

# ============================================
# Configuration
# ============================================
CBR_PATH = "C:/Users/Delta/Documents/datasetML/chatbot/chat/dataforCBR"
LOG_FILE = os.path.join(CBR_PATH, "feedback_log.txt")
CASE_FILE = os.path.join(CBR_PATH, "case_base_data.pickle")
VECTORIZER_FILE = os.path.join(CBR_PATH, "tfidf_vectorizer.pickle")

# Verify paths exist
try:
    if not os.path.exists(CBR_PATH):
        logging.error(f"CBR path does not exist: {CBR_PATH}")
        raise FileNotFoundError(f"CBR path not found: {CBR_PATH}")
    logging.info(f"✓ CBR path verified: {CBR_PATH}")
except Exception as e:
    logging.error(f"Path verification failed: {e}")
    print(f"ERROR: {e}")

# ============================================
# Logging Function
# ============================================
def log_feedback(user_input, bot_response, model_used, feedback_value=None):
    """
    Logs every interaction into a feedback log file
    
    Args:
        user_input (str): User's original input
        bot_response (str): Bot's response
        model_used (str): Which model made the prediction (RNN, CBR, etc.)
        feedback_value (str, optional): User's feedback (good/bad/neutral)
    
    Returns:
        bool: True if logging successful, False otherwise
    """
    try:
        # Validate inputs
        if not user_input or not isinstance(user_input, str):
            logging.warning("Invalid user_input in log_feedback")
            return False
        
        if not bot_response or not isinstance(bot_response, str):
            logging.warning("Invalid bot_response in log_feedback")
            return False
        
        # Create log entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] Model: {model_used} | Feedback: {feedback_value} | User: {user_input[:100]} | Bot: {bot_response[:100]}\n"
        
        # Write to file
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry)
        
        logging.info(f"Logged feedback: {model_used}")
        return True
    
    except IOError as e:
        logging.error(f"Failed to write to feedback log: {e}")
        print(f"ERROR: Could not write feedback log: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error in log_feedback: {e}")
        return False

# ============================================
# Case Retention Function
# ============================================
def retain_new_case(user_input, correct_answer, priority="User_Feedback"):
    """
    Adds a new case to the CBR case base when user corrects the chatbot
    
    Args:
        user_input (str): The original problem description
        correct_answer (str): The correct department/solution
        priority (str): Priority level (default: "User_Feedback")
    
    Returns:
        bool: True if case added successfully, False otherwise
    """
    try:
        # Input validation
        if not user_input or not isinstance(user_input, str):
            logging.warning("Invalid user_input in retain_new_case")
            return False
        
        if not correct_answer or not isinstance(correct_answer, str):
            logging.warning("Invalid correct_answer in retain_new_case")
            return False
        
        # Load current case base
        try:
            with open(CASE_FILE, "rb") as f:
                case_base = pickle.load(f)
            logging.info("✓ Current case base loaded")
        except FileNotFoundError:
            logging.error(f"Case base file not found: {CASE_FILE}")
            print(f"ERROR: Case base not found. Run preprocessing first.")
            return False
        except Exception as e:
            logging.error(f"Failed to load case base: {e}")
            print(f"ERROR: Failed to load case base: {e}")
            return False
        
        # Extract current data
        try:
            vectors = case_base["vectors"]
            bodies = list(case_base["body_text"])
            solutions = list(case_base["solutions"])
            priorities = list(case_base["priority"])
            
            logging.info(f"Current case base size: {len(bodies)}")
        except KeyError as e:
            logging.error(f"Invalid case base structure: {e}")
            print(f"ERROR: Case base structure invalid: {e}")
            return False
        
        # Load TF-IDF vectorizer
        try:
            with open(VECTORIZER_FILE, "rb") as f:
                tfidf_vectorizer = pickle.load(f)
            logging.info("✓ TF-IDF vectorizer loaded")
        except FileNotFoundError:
            logging.error(f"Vectorizer not found: {VECTORIZER_FILE}")
            print(f"ERROR: Vectorizer not found.")
            return False
        except Exception as e:
            logging.error(f"Failed to load vectorizer: {e}")
            return False
        
        # Vectorize new case
        try:
            new_vec = tfidf_vectorizer.transform([user_input])
            
            # Check dimension compatibility
            if new_vec.shape[1] != vectors.shape[1]:
                logging.error(f"Vector dimension mismatch: {new_vec.shape[1]} vs {vectors.shape[1]}")
                print(f"ERROR: Vector dimension mismatch.")
                return False
            
            logging.info("✓ New case vectorized successfully")
        except Exception as e:
            logging.error(f"Failed to vectorize new case: {e}")
            print(f"ERROR: Failed to vectorize: {e}")
            return False
        
        # Add new case to case base
        try:
            # Append text data
            bodies.append(user_input)
            solutions.append(correct_answer)
            priorities.append(priority)
            
            # Append vector (convert to dense if needed, then vstack)
            vectors = vstack([vectors, new_vec])
            
            logging.info(f"New case added. Case base size now: {len(bodies)}")
        except Exception as e:
            logging.error(f"Failed to add case to base: {e}")
            print(f"ERROR: Failed to add case: {e}")
            return False
        
        # Save updated case base
        try:
            with open(CASE_FILE, "wb") as f:
                pickle.dump({
                    "vectors": vectors,
                    "body_text": bodies,
                    "solutions": solutions,
                    "priority": priorities
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logging.info("✓ Updated case base saved")
            return True
        except IOError as e:
            logging.error(f"Failed to save case base: {e}")
            print(f"ERROR: Failed to save case base: {e}")
            return False
    
    except Exception as e:
        logging.error(f"Unexpected error in retain_new_case: {e}")
        print(f"ERROR: Unexpected error: {e}")
        return False

# ============================================
# User Feedback Handler
# ============================================
def user_feedback(is_correct, user_input, bot_response, model_used="UNKNOWN"):
    """
    Handles user feedback and updates CBR if needed
    
    Args:
        is_correct (bool): Whether the bot's response was correct
        user_input (str): Original user input
        bot_response (str): Bot's response
        model_used (str): Which model made the prediction
    
    Returns:
        bool: True if feedback processed successfully
    """
    try:
        if is_correct:
            # Log as correct
            success = log_feedback(user_input, bot_response, model_used, feedback_value="CORRECT")
            if success:
                print("✔️ Feedback recorded as correct.")
                logging.info("Feedback recorded as correct")
                return True
            else:
                print("⚠️ Warning: Feedback could not be fully logged.")
                return False
        
        else:
            # Ask for correction
            print("\n❌ Feedback registered as incorrect.")
            print("Please provide the correct response:")
            
            try:
                correct_answer = input("Correct Response: ").strip()
                
                if not correct_answer:
                    print("⚠️ No response provided. Feedback not saved.")
                    logging.warning("User provided empty correction")
                    return False
                
                # Add new case to CBR
                case_added = retain_new_case(user_input, correct_answer, priority="User_Correction")
                
                # Log the feedback
                log_success = log_feedback(user_input, correct_answer, model_used, 
                                         feedback_value="CORRECTED")
                
                if case_added and log_success:
                    print("✅ New case added to CBR dataset and logged successfully.")
                    logging.info(f"User correction processed: {correct_answer}")
                    return True
                else:
                    print("⚠️ Feedback recorded but case addition may have failed.")
                    return False
            
            except KeyboardInterrupt:
                print("\n⚠️ Feedback cancelled by user.")
                logging.info("User cancelled feedback")
                return False
            except Exception as e:
                logging.error(f"Error processing user correction: {e}")
                print(f"⚠️ Error processing correction: {e}")
                return False
    
    except Exception as e:
        logging.error(f"Unexpected error in user_feedback: {e}")
        print(f"ERROR: {e}")
        return False

# ============================================
# Get Feedback Statistics
# ============================================
def get_feedback_statistics():
    """
    Retrieve feedback statistics from log file
    
    Returns:
        dict: Statistics about feedback collected
    """
    try:
        if not os.path.exists(LOG_FILE):
            logging.warning("Feedback log file does not exist")
            return {
                "total_feedbacks": 0,
                "correct_count": 0,
                "incorrect_count": 0,
                "success_rate": 0.0
            }
        
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        total = len(lines)
        correct = sum(1 for line in lines if "CORRECT" in line or "Feedback: good" in line)
        incorrect = sum(1 for line in lines if "INCORRECT" in line or "CORRECTED" in line)
        
        stats = {
            "total_feedbacks": total,
            "correct_count": correct,
            "incorrect_count": incorrect,
            "success_rate": (correct / total * 100) if total > 0 else 0.0
        }
        
        logging.info(f"Feedback statistics: {stats}")
        return stats
    
    except Exception as e:
        logging.error(f"Failed to get feedback statistics: {e}")
        return {}

# ============================================
# Example Usage
# ============================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FEEDBACK SYSTEM DEMO")
    print("=" * 80)
    
    # Test logging
    print("\n1. Testing feedback logging...")
    try:
        test_logged = log_feedback(
            user_input="My VPN is not working",
            bot_response="Technical Support",
            model_used="RNN+CBR",
            feedback_value="CORRECT"
        )
        print(f"   ✓ Logging test: {'PASSED' if test_logged else 'FAILED'}")
    except Exception as e:
        print(f"   ✗ Logging test failed: {e}")
    
    # Display statistics
    print("\n2. Feedback Statistics:")
    try:
        stats = get_feedback_statistics()
        print(f"   Total Feedbacks: {stats.get('total_feedbacks', 0)}")
        print(f"   Correct: {stats.get('correct_count', 0)}")
        print(f"   Incorrect: {stats.get('incorrect_count', 0)}")
        print(f"   Success Rate: {stats.get('success_rate', 0):.1f}%")
    except Exception as e:
        print(f"   ✗ Could not retrieve statistics: {e}")
    
    print("\n" + "=" * 80)
    print("✓ Feedback System Demo Completed")
    print("=" * 80)