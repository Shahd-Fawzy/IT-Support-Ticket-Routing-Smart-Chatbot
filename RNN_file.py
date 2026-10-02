import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, Bidirectional, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# 1. Define the path 
DATA_PATH = "C:/Users/Delta/Documents/datasetML/chatbot/chat/dataforRNN"

print(f"Loading data from: {DATA_PATH}")

# 2. Load the data (assuming preprocessing is done)
X_train = np.load(os.path.join(DATA_PATH, 'X_train_pad.npy'))
y_train = np.load(os.path.join(DATA_PATH, 'y_train.npy'))
X_val = np.load(os.path.join(DATA_PATH, 'X_val_pad.npy'))
y_val = np.load(os.path.join(DATA_PATH, 'y_val.npy'))
X_test = np.load(os.path.join(DATA_PATH, 'X_test_pad.npy'))
y_test = np.load(os.path.join(DATA_PATH, 'y_test.npy'))

with open(os.path.join(DATA_PATH, 'tokenizer.pickle'), 'rb') as f:
    tokenizer = pickle.load(f)
with open(os.path.join(DATA_PATH, 'label_encoder.pickle'), 'rb') as f:
    label_enc = pickle.load(f)

print("Data Loaded Successfully!")
print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
print(f"Class distribution: {np.bincount(y_train)}")

# 3. Network Hyperparameters
vocab_size = len(tokenizer.word_index) + 1
max_length = X_train.shape[1]
num_classes = len(label_enc.classes_)
embedding_dim = 128

# IMPROVEMENT 1: Compute class weights to handle imbalanced data
class_weights = compute_class_weight('balanced', 
                                     classes=np.unique(y_train), 
                                     y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print(f"Class weights: {class_weight_dict}")

# 4. Building the IMPROVED Anti-Overfitting Model Architecture
model = Sequential([
    # Layer 1: Embedding with L2 regularization
    Embedding(input_dim=vocab_size, 
              output_dim=embedding_dim, 
              input_length=max_length,
              embeddings_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    
    # IMPROVEMENT 2: Embedding Dropout (prevents co-adaptation in embedding space)
    Dropout(0.3),
    
    # Layer 2: Bidirectional GRU with strong regularization
    Bidirectional(GRU(64, 
                      return_sequences=True, 
                      dropout=0.4, 
                      recurrent_dropout=0.4,
                      kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                      recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-4))), 
    
    # Layer 3: Global Max Pooling
    tf.keras.layers.GlobalMaxPooling1D(),
    
    # Layer 4: Layer Normalization
    LayerNormalization(),

    # IMPROVEMENT 3: Stacked Dense layers with progressive dropout
    Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)), 
    Dropout(0.5),  # Increased dropout
    
    Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    Dropout(0.4),
    
    Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    Dropout(0.3),
    
    # Output Layer
    Dense(num_classes, activation='softmax') 
])

# IMPROVEMENT 4: Conservative optimizer settings
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0003,  # Reduced learning rate
    global_clipnorm=1.0  # Gradient clipping by norm (choose only one)
)

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer=optimizer, 
              metrics=['accuracy'])

print("\n" + "="*80)
model.summary()
print("="*80)

# 5. Training Callbacks (Aggressive Early Stopping)
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5,  # Slightly more patience to avoid premature stopping
    restore_best_weights=True, 
    min_delta=0.001,  # Increase in delta for stability
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5,  # Reduce LR by 50%
    patience=3, 
    min_lr=0.00001,
    verbose=1
)

checkpoint_path = os.path.join(DATA_PATH, 'best_rnn_model.h5')
checkpoint = ModelCheckpoint(
    checkpoint_path, 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max',
    verbose=1
)

print("\nStarting Training with Overfitting Prevention Strategies...")
print("- Class weighting for imbalanced data")
print("- L1/L2 regularization on all layers")
print("- Progressive dropout (0.3 to 0.5)")
print("- Learning rate scheduling")
print("- Early stopping with patience=5\n")

# 6. Start Training
history = model.fit(
    X_train, y_train,
    epochs=50,  # Increased epochs (early stopping will handle it)
    batch_size=16,  # Smaller batch size for better regularization
    validation_data=(X_val, y_val),
    callbacks=[early_stop, checkpoint, reduce_lr],
    class_weight=class_weight_dict,  # IMPROVEMENT 5: Handle class imbalance
    verbose=1
)

print("\n" + "="*80)
print("Training Completed.")
print("="*80)

# 7. Final Evaluation (Best Model)
best_model = tf.keras.models.load_model(checkpoint_path)
print("\nEvaluating on Test Data (Best Checkpoint Model)...")
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Calculate train accuracy for comparison
train_loss, train_acc = best_model.evaluate(X_train, y_train, verbose=0)
print(f"\nTrain Accuracy: {train_acc*100:.2f}%")
print(f"Train Loss: {train_loss:.4f}")
print(f"Overfitting Gap: {(train_acc - test_acc)*100:.2f}% (should be < 10%)")

# Detailed classification report
y_pred_probs = best_model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_enc.classes_))

# 8. Save Final Model
final_model_path = os.path.join(DATA_PATH, 'rnn_model_final.h5')
best_model.save(final_model_path)
print(f"\nBest Model Saved at: {final_model_path}")

# 9. Analyze Overfitting with Detailed Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0, 0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Epochs')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Loss
axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 1].set_title('Model Loss', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Epochs')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Accuracy Difference (Overfitting indicator)
accuracy_diff = np.array(history.history['accuracy']) - np.array(history.history['val_accuracy'])
axes[1, 0].plot(accuracy_diff, label='Train - Val Accuracy', color='red', linewidth=2)
axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1, 0].fill_between(range(len(accuracy_diff)), accuracy_diff, alpha=0.3, color='red')
axes[1, 0].set_title('Overfitting Indicator (Train-Val Gap)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Epochs')
axes[1, 0].set_ylabel('Accuracy Difference')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Learning Rate Schedule (if available)
if hasattr(history, 'epoch'):
    axes[1, 1].text(0.5, 0.5, f'Best Epoch: {np.argmin(history.history["val_loss"])}\n' +
                               f'Final Val Loss: {history.history["val_loss"][-1]:.4f}\n' +
                               f'Test Accuracy: {test_acc*100:.2f}%\n' +
                               f'Overfitting Gap: {(train_acc - test_acc)*100:.2f}%',
                    ha='center', va='center', fontsize=11, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    family='monospace')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, 'training_analysis.png'), dpi=300, bbox_inches='tight')
print(f"\nTraining analysis plot saved to: {os.path.join(DATA_PATH, 'training_analysis.png')}")
plt.show()

print("\n" + "="*80)
print("TRAINING SUMMARY")
print("="*80)
print(f"✓ Early stopping patience: 5 epochs")
print(f"✓ L1/L2 regularization applied to all layers")
print(f"✓ Dropout progression: 0.3 → 0.5 → 0.4 → 0.3")
print(f"✓ Class weights applied to handle imbalance")
print(f"✓ Learning rate: 0.0003 with scheduling")
print(f"✓ Gradient clipping: global_clipnorm=1.0")
print("="*80)