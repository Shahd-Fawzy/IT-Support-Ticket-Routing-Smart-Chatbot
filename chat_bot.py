import os
import pickle
import numpy as np
import re
import nltk
import logging
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from collections import Counter
import json # لإمكانية التعامل مع json في المستقبل

# ============================================
# LOGGING SETUP
# ============================================
logging.basicConfig(
    filename='chatbot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("=" * 80)
logging.info("CHATBOT STARTED")
logging.info("=" * 80)

# ============================================
# Import custom modules
# ============================================
try:
    from cbr_feedback import log_feedback, user_feedback, get_feedback_statistics
    from preprocessing_input import validate_input, clean_text, preprocess_text
    # يتم استيراد cbr_retrieve و get_cbr_statistics من هذا الملف
    from cbr_retrieval import retrieve_similar_cases, get_cbr_statistics 
    logging.info("✓ All custom modules imported successfully")
except ImportError as e:
    logging.error(f"✗ Failed to import custom modules: {e}")
    print(f"ERROR: Failed to import custom modules: {e}")
    print("Make sure all module files are in the correct directory.")
    exit(1)

# ============================================
# Configuration
# ============================================
RNN_PATH = "C:/Users/Delta/Documents/datasetML/chatbot/chat/dataforRNN"
CBR_PATH = "C:/Users/Delta/Documents/datasetML/chatbot/chat/dataforCBR"

# RNN Model Confidence Threshold
RNN_CONFIDENCE_THRESHOLD = 0.5

# CBR Similarity Threshold
CBR_SIMILARITY_THRESHOLD = 0.2

# ============================================
# Load RNN Model
# ============================================
print("Loading RNN Model...")
logging.info("Loading RNN model...")

try:
    tokenizer = pickle.load(open(f"{RNN_PATH}/tokenizer.pickle", "rb"))
    label_enc = pickle.load(open(f"{RNN_PATH}/label_encoder.pickle", "rb"))
    rnn_model = load_model(f"{RNN_PATH}/rnn_model_final.h5")
    
    # Load metadata (for max_length)
    try:
        with open(f"{RNN_PATH}/metadata.pickle", "rb") as f:
            metadata = pickle.load(f)
        max_length = metadata.get('max_length', 100)
    except:
        max_length = 100
        logging.warning("Could not load metadata, using default max_length")
    
    print("✓ RNN model loaded successfully")
    logging.info("✓ RNN model loaded")
    
except FileNotFoundError as e:
    logging.error(f"✗ RNN model files not found: {e}")
    print(f"ERROR: RNN model not found. Make sure rnn_training.py has been run.")
    exit(1)
except Exception as e:
    logging.error(f"✗ Failed to load RNN model: {e}")
    print(f"ERROR: Failed to load RNN model: {e}")
    exit(1)

# ============================================
# Load CBR System Check (NO NEED TO LOAD DATA HERE - Done in cbr_retrieval.py)
# ============================================
# تم حذف كود تحميل قاعدة بيانات CBR هنا، ونعتمد على تحميله في cbr_retrieval.py

# ============================================
# RNN Prediction Function (No Change)
# ============================================
def rnn_predict(text, confidence_threshold=RNN_CONFIDENCE_THRESHOLD):
    """Predict department using RNN model."""
    try:
        # Tokenize and pad
        seq = tokenizer.texts_to_sequences([text])
        pad = pad_sequences(seq, maxlen=max_length, padding="post")
        
        # Predict
        pred = rnn_model.predict(pad, verbose=0)
        confidence = np.max(pred)
        class_id = np.argmax(pred)
        
        # Check confidence threshold
        if confidence < confidence_threshold:
            logging.info(f"RNN confidence below threshold: {confidence:.3f}")
            return None, confidence
        
        department = label_enc.inverse_transform([class_id])[0]
        logging.info(f"RNN prediction: {department} (confidence: {confidence:.3f})")
        
        return department, confidence
    
    except Exception as e:
        logging.error(f"Error in RNN prediction: {e}")
        print(f"ERROR in RNN prediction: {e}")
        return None, None

# ============================================
# CBR Retrieval Function (Uses imported function)
# ============================================
def cbr_retrieve(text, k=3, min_similarity=CBR_SIMILARITY_THRESHOLD):
    """Retrieve similar cases from CBR system using imported function."""
    try:
        # نعتمد على الدالة المستوردة من cbr_retrieval.py
        results = retrieve_similar_cases(text, k=k, min_similarity=min_similarity)
        logging.info(f"CBR retrieved {len(results)} cases")
        return results
    except Exception as e:
        logging.error(f"Error in CBR retrieval: {e}")
        print(f"ERROR in CBR retrieval: {e}")
        return []


# ============================================
# ✓ NEW FUNCTION: Extract Solution Hints from Tags
# ============================================
def get_combined_solution_hints(cbr_cases, top_n=3):
    """
    Extracts the most frequent Tags (Solution Hints) and the most frequent Priority 
    from the retrieved CBR cases.
    
    Args:
        cbr_cases (list): List of similar cases.
        top_n (int): Number of top tags to return.
        
    Returns:
        tuple: (list of top tags, most frequent priority)
    """
    all_tags = []
    all_priorities = []
    
    for case in cbr_cases:
        # تأكد من أن tags هي قائمة، وإلا تخطى الحالة
        if isinstance(case.get('tags'), list):
            all_tags.extend(case['tags'])
        
        all_priorities.append(case['priority'])

    # حساب العلامات (Tags) الأكثر تكراراً
    tag_counts = Counter(all_tags)
    top_tags = [tag for tag, count in tag_counts.most_common(top_n)]
    
    # حساب الأولوية الأكثر تكراراً
    priority_counts = Counter(all_priorities)
    most_freq_priority = priority_counts.most_common(1)[0][0] if priority_counts else "Unknown"

    return top_tags, most_freq_priority

# ============================================
# Response Ranking Function (Now incorporates Solution Hints)
# ============================================
def get_best_response(rnn_dept, rnn_conf, cbr_cases):
    """
    Rank responses and select best one based on confidence/similarity.
    """
    # استخلاص Solution Hints والأولوية
    solution_hints, predicted_priority = get_combined_solution_hints(cbr_cases)
    
    # اذا كان RNN واثقاً جداً
    if rnn_dept is not None and rnn_conf > 0.7:
        best_dept = rnn_dept
        source = 'RNN'
        confidence = rnn_conf
        explanation = f'High confidence RNN prediction confirmed by similar cases.'
    
    # اذا كان CBR لديه تطابق عالٍ
    elif cbr_cases and cbr_cases[0]['similarity'] > 0.6:
        # نحصل على القسم الأكثر شيوعاً في CBR
        cbr_depts = [c['department'] for c in cbr_cases]
        best_dept = Counter(cbr_depts).most_common(1)[0][0]
        source = 'CBR'
        confidence = cbr_cases[0]['similarity']
        explanation = f'High similarity case found, department based on CBR consensus.'
    
    # اذا كان RNN واثقاً بدرجة متوسطة
    elif rnn_dept is not None and rnn_conf > 0.5:
        best_dept = rnn_dept
        source = 'RNN'
        confidence = rnn_conf
        explanation = f'RNN prediction (moderate confidence).'
        
    # إذا لم يكن هناك قسم واثق لكن يوجد دليل CBR
    elif cbr_cases:
        cbr_depts = [c['department'] for c in cbr_cases]
        best_dept = Counter(cbr_depts).most_common(1)[0][0]
        source = 'CBR (Low Sim)'
        confidence = cbr_cases[0]['similarity']
        explanation = f'No high confidence prediction. Suggestion based on best available case.'
        
    # لا يوجد تنبؤ واثق
    else:
        best_dept = None
        source = 'UNCERTAIN'
        confidence = 0.0
        explanation = 'Unable to confidently determine department. Please contact support.'
        
    return {
        'source': source,
        'department': best_dept,
        'confidence': confidence,
        'explanation': explanation,
        'solution_hints': solution_hints,         # ✓ إضافة Solution Hints
        'predicted_priority': predicted_priority  # ✓ إضافة الأولوية المتوقعة
    }

# ============================================
# Main Chatbot Reply Function (No Change in logic flow)
# ============================================
def chatbot_reply(user_input):
    """Main chatbot logic - processes input and returns response"""
    try:
        # Step 1 & 2: Validate and Preprocess input (Uses imported functions)
        is_valid, error = validate_input(user_input)
        if not is_valid:
            logging.warning(f"Input validation failed: {error}")
            print(f"⚠️ {error}")
            return None, None
        
        preprocessed = preprocess_text(user_input, verbose=False)
        if not preprocessed['is_valid']:
            logging.warning(f"Preprocessing failed: {preprocessed['error']}")
            print(f"⚠️ {preprocessed['error']}")
            return None, None
        
        cleaned_text = preprocessed['cleaned']
        
        # Step 3: Get RNN prediction
        rnn_dept, rnn_conf = rnn_predict(cleaned_text)
        
        # Step 4: Get CBR cases
        cbr_cases = cbr_retrieve(cleaned_text, k=3, min_similarity=CBR_SIMILARITY_THRESHOLD)
        
        # Step 5: Rank and select best response
        best_response = get_best_response(rnn_dept, rnn_conf if rnn_conf else 0, cbr_cases)
        
        return best_response, cbr_cases
    
    except Exception as e:
        logging.error(f"Error in chatbot_reply: {e}")
        print(f"ERROR: {e}")
        return None, None

# ============================================
# Display Response Function
# ============================================
def display_response(best_response, cbr_cases):
    """
    Display chatbot response in formatted way
    """
    try:
        if not best_response:
            print("❌ Could not process request.")
            return
        
        print("\n" + "=" * 80)
        print("🤖 CHATBOT RESPONSE (Tier-1 Support)")
        print("=" * 80)
        
        # Display best prediction
        print(f"\n📌 PRIMARY RECOMMENDATION:")
        print(f"  Department: {best_response['department'] or 'Unknown'}")
        print(f"  Source: {best_response['source']} | Confidence: {best_response['confidence']*100:.1f}%")
        print(f"  Priority: {best_response['predicted_priority']}") # ✓ عرض الأولوية المتوقعة
        print(f"  Explanation: {best_response['explanation']}")

        # ✓ عرض تلميحات الحل المستخلصة من Tags
        hints = best_response['solution_hints']
        if hints:
            print(f"\n💡 SOLUTION HINTS (Checkpoints):")
            print(f"  Focus on these areas: {', '.join(hints)}")
        
        # Display CBR cases if available
        if cbr_cases:
            print(f"\n📚 SUPPORTING EVIDENCE (CBR Cases):")
            for idx, case in enumerate(cbr_cases, 1):
                # نستخدم .get لتجنب الانهيار إذا لم يكن حقل tags موجودًا لأي سبب
                tags_str = ", ".join(case.get('tags', []))
                
                print(f"\n  Case {idx} ({case['department']}):")
                print(f"    Similarity: {case['similarity']:.3f}")
                print(f"    Priority: {case['priority']}")
                print(f"    Tags: {tags_str}")
                print(f"    Text: {case['body'][:100]}...")
        else:
            print(f"\n📚 Supporting Evidence: None found")
        
        print("\n" + "=" * 80)
    
    except Exception as e:
        logging.error(f"Error displaying response: {e}")
        print(f"ERROR displaying response: {e}")

# ============================================
# User Feedback Collection (No Change in logic)
# ============================================
def collect_feedback(user_input, best_response):
    # ... (الكود لا يتغير)
    """Collect user feedback on the chatbot response"""
    try:
        if best_response['source'] == 'ERROR' or best_response['source'] == 'UNCERTAIN':
            return False
        
        print("\n" + "-" * 80)
        feedback = input("Was this prediction helpful? (yes/no): ").strip().lower()
        
        if feedback in ["yes", "y", "ok", "good", "correct"]:
            is_correct = True
        elif feedback in ["no", "n", "wrong", "bad", "incorrect"]:
            is_correct = False
        else:
            print("⚠️ Invalid input. Skipping feedback.")
            return False
        
        # Process feedback (Uses imported function)
        success = user_feedback(
            is_correct=is_correct,
            user_input=user_input,
            bot_response=best_response['department'] or 'Unknown',
            model_used=best_response['source']
        )
        
        return success
    
    except KeyboardInterrupt:
        print("\n⚠️ Feedback skipped by user.")
        return False
    except Exception as e:
        logging.error(f"Error collecting feedback: {e}")
        print(f"⚠️ Error processing feedback: {e}")
        return False


# ============================================
# Main CLI Loop (No Change in logic)
# ============================================
def main():
    """Main chatbot CLI loop"""
    # ... (الكود لا يتغير)
    print("\n" + "=" * 80)
    print("🤖 IT SUPPORT TICKET CHATBOT")
    print("=" * 80)
    print("\nWelcome! I can help you route your IT support issue to the right department.")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("Type 'stats' to see feedback statistics.")
    print("=" * 80)
    
    logging.info("Chatbot CLI started")
    interaction_count = 0
    
    while True:
        try:
            user_input = input("\n📝 Describe your issue: ").strip()
            
            # Handle special commands
            if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                print("\n👋 Thank you for using the chatbot! Goodbye!")
                logging.info(f"Chatbot session ended. Interactions: {interaction_count}")
                break
            
            if user_input.lower() == "stats":
                try:
                    stats = get_feedback_statistics()
                    print(f"\n📊 Feedback Statistics:")
                    print(f"  Total Feedbacks: {stats.get('total_feedbacks', 0)}")
                    print(f"  Correct: {stats.get('correct_count', 0)}")
                    print(f"  Incorrect: {stats.get('incorrect_count', 0)}")
                    print(f"  Success Rate: {stats.get('success_rate', 0):.1f}%")
                except Exception as e:
                    print(f"Could not retrieve statistics: {e}")
                continue
            
            if not user_input:
                print("⚠️ Please enter a valid issue description.")
                continue
            
            # Get chatbot response
            print("\n🔄 Processing...")
            best_response, cbr_cases = chatbot_reply(user_input)
            
            if best_response:
                # Display response
                display_response(best_response, cbr_cases)
                
                # Collect feedback
                collect_feedback(user_input, best_response)
                interaction_count += 1
            else:
                print("❌ Failed to process request. Please try again.")
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            logging.info("Chatbot interrupted by user")
            break
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}")
            print(f"❌ An unexpected error occurred: {e}")
            print("Please try again or contact support.")

# ============================================
# Entry Point (No Change)
# ============================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Critical error: {e}")
        print(f"CRITICAL ERROR: {e}")
        exit(1)