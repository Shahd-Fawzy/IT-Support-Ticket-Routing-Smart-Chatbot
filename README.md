# 🤖 Intelligent Hybrid IT Support Ticket Chatbot

An end-to-end intelligent chatbot system designed to automate IT support ticket routing. This system utilizes a **Hybrid Architecture** combining a Deep Learning sequence classifier (Bidirectional GRU) with **Case-Based Reasoning (CBR)** using TF-IDF and cosine similarity. It also features a self-improving feedback loop that dynamically appends user corrections to the case base.

## 🚀 Key Features
* **Dual-Engine Routing:** Combines neural network intent prediction with historical case matching.
* **Advanced Text Preprocessing:** Custom cleaning pipeline handling stopword removal, tokenization, padding, and text normalization.
* **Deep Learning Classifier:** Built using TensorFlow/Keras with Bidirectional GRU layers, SpatialDropout1D, class weighting for imbalanced datasets, and advanced regularization.
* **Dynamic Case Retention:** Captures incorrect predictions at runtime, prompts users for corrections, and automatically retains new cases to enhance future accuracy.
* **Comprehensive Logging:** Robust logging system tracking user inputs, system confidence, and feedback statistics.

## 📂 Project Structure
```text
├── preprossing_intelligent.py   # Dataset processing, tokenization, & TF-IDF vectorization
├── RNN_file.py                  # Bidirectional GRU model training script
├── gru_training.py              # Advanced GRU model optimization pipeline
├── cbr_model.py                 # Case-Based Reasoning retrieval engine
├── cbr_feedback.py              # Runtime user feedback & case-retention handler
├── chat_bot.py                  # Main interactive CLI application loop
└── preprocessing_input.py       # Input validation and text utility modules
