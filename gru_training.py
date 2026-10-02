import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, Bidirectional, LayerNormalization, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LambdaCallback
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import classification_report, f1_score
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

# 4. Building IMPROVED Model - Enhanced Capacity with Regularization
# Strategy: Balance model capacity with strong regularization
model = Sequential([
    # Layer 1: Embedding with L2 regularization
    Embedding(input_dim=vocab_size, 
              output_dim=embedding_dim, 
              input_length=max_length,
              embeddings_regularizer=l1_l2(l1=1e-6, l2=1e-5)),  # Lighter L2 for better embeddings
    
    # IMPROVEMENT 2: SpatialDropout1D (better for sequential data than standard Dropout)
    SpatialDropout1D(0.2),
    
    # Layer 2: FIRST Bidirectional GRU - Increased capacity (128 units)
    Bidirectional(GRU(128, 
                      return_sequences=True, 
                      dropout=0.3, 
                      recurrent_dropout=0.3,
                      kernel_regularizer=l1_l2(l1=1e-6, l2=1e-5),
                      recurrent_regularizer=l1_l2(l1=1e-6, l2=1e-5))),
    
    # IMPROVEMENT 3: Second Bidirectional GRU Layer (stacking for better feature learning)
    Bidirectional(GRU(64, 
                      return_sequences=True, 
                      dropout=0.3, 
                      recurrent_dropout=0.3,
                      kernel_regularizer=l1_l2(l1=1e-6, l2=1e-5),
                      recurrent_regularizer=l1_l2(l1=1e-6, l2=1e-5))),
    
    # Layer 3: Global Max Pooling
    tf.keras.layers.GlobalMaxPooling1D(),
    
    # Layer 4: Layer Normalization
    LayerNormalization(),

    # IMPROVEMENT 4: Increased Dense Layer Capacity with Balanced Regularization
    Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-6, l2=1e-5)), 
    Dropout(0.4),
    
    Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-6, l2=1e-5)),
    Dropout(0.3),
    
    Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-6, l2=1e-5)),
    Dropout(0.2),
    
    # Output Layer
    Dense(num_classes, activation='softmax') 
])

# IMPROVEMENT 5: Optimized optimizer with better learning rate
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,  # Slightly increased for faster convergence
    beta_1=0.9,           # Standard momentum
    beta_2=0.999,         # Standard RMSprop
    epsilon=1e-7,
    global_clipnorm=1.0
)

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer=optimizer, 
              metrics=['accuracy'])

print("\n" + "="*80)
model.summary()
print("="*80)

# 5. Training Callbacks - Optimized for Accuracy
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=8,  # More patience to let model train longer
    restore_best_weights=True, 
    min_delta=0.0005,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.7,  # Reduce LR more gradually
    patience=4, 
    min_lr=0.00001,
    verbose=1
)

checkpoint_path = os.path.join(DATA_PATH, 'best_rnn_model.h5')
checkpoint = ModelCheckpoint(
    checkpoint_path, 
    monitor='val_accuracy',  # Monitor accuracy instead of loss
    save_best_only=True, 
    mode='max',
    verbose=1
)

# IMPROVEMENT 6: Custom callback to monitor overfitting
class OverfittingMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        gap = train_acc - val_acc
        if gap > 0.15:  # Alert if gap exceeds 15%
            print(f"\n⚠️  WARNING: High overfitting gap detected: {gap*100:.2f}%")

print("\nStarting Training with ACCURACY IMPROVEMENTS...")
print("- Increased model capacity (dual GRU layers, larger Dense layers)")
print("- Optimized regularization (lighter L2 for better learning)")
print("- SpatialDropout1D for sequential data")
print("- Extended patience (8 epochs) for better convergence")
print("- Learning rate: 0.001 with scheduling")
print("- Class weighting for imbalanced data\n")

# 6. Start Training
history = model.fit(
    X_train, y_train,
    epochs=80,  # More epochs with early stopping
    batch_size=32,  # Larger batch size for stability (helps accuracy)
    validation_data=(X_val, y_val),
    callbacks=[early_stop, checkpoint, reduce_lr, OverfittingMonitor()],
    class_weight=class_weight_dict,
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
print(f"Overfitting Gap: {(train_acc - test_acc)*100:.2f}% (target: < 10%)")

# Detailed classification report
y_pred_probs = best_model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_enc.classes_))

# F1-Score Summary
f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
print(f"\nF1-Score (Weighted): {f1_weighted:.4f}")
print(f"F1-Score (Macro): {f1_macro:.4f}")

# 8. Save Final Model
final_model_path = os.path.join(DATA_PATH, 'rnn_model_final.h5')
best_model.save(final_model_path)
print(f"\nBest Model Saved at: {final_model_path}")

# 9. Comprehensive Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

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

# Plot 3: Accuracy Difference
accuracy_diff = np.array(history.history['accuracy']) - np.array(history.history['val_accuracy'])
axes[0, 2].plot(accuracy_diff, label='Train - Val Accuracy', color='red', linewidth=2)
axes[0, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[0, 2].axhline(y=0.10, color='orange', linestyle='--', alpha=0.5, label='10% threshold')
axes[0, 2].fill_between(range(len(accuracy_diff)), accuracy_diff, alpha=0.3, color='red')
axes[0, 2].set_title('Overfitting Indicator', fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('Epochs')
axes[0, 2].set_ylabel('Accuracy Gap')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Validation Accuracy over Epochs
axes[1, 0].plot(history.history['val_accuracy'], color='green', linewidth=3, marker='o', markersize=4)
axes[1, 0].set_title('Validation Accuracy Progression', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Epochs')
axes[1, 0].set_ylabel('Validation Accuracy')
axes[1, 0].grid(True, alpha=0.3)
best_epoch = np.argmax(history.history['val_accuracy'])
best_val_acc = history.history['val_accuracy'][best_epoch]
axes[1, 0].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best: {best_val_acc*100:.2f}%')
axes[1, 0].legend()

# Plot 5: Learning Rate Schedule (if available)
if hasattr(history, 'epoch'):
    axes[1, 1].text(0.5, 0.5, f'🎯 BEST EPOCH: {best_epoch + 1}\n\n' +
                               f'Val Accuracy: {best_val_acc*100:.2f}%\n' +
                               f'Test Accuracy: {test_acc*100:.2f}%\n' +
                               f'Overfitting Gap: {(train_acc - test_acc)*100:.2f}%\n\n' +
                               f'F1 (Weighted): {f1_weighted:.4f}\n' +
                               f'F1 (Macro): {f1_macro:.4f}',
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                    family='monospace')
axes[1, 1].axis('off')

# Plot 6: Train vs Val Loss Ratio
loss_ratio = np.array(history.history['loss']) / (np.array(history.history['val_loss']) + 1e-8)
axes[1, 2].plot(loss_ratio, color='purple', linewidth=2)
axes[1, 2].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Ideal (1.0)')
axes[1, 2].set_title('Train/Val Loss Ratio', fontsize=12, fontweight='bold')
axes[1, 2].set_xlabel('Epochs')
axes[1, 2].set_ylabel('Ratio')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, 'training_analysis.png'), dpi=300, bbox_inches='tight')
print(f"\nTraining analysis plot saved to: {os.path.join(DATA_PATH, 'training_analysis.png')}")
plt.show()

print("\n" + "="*80)
print("TRAINING SUMMARY - ACCURACY IMPROVEMENTS")
print("="*80)
print(f"✓ Dual Bidirectional GRU layers (128 + 64 units)")
print(f"✓ Increased Dense layers (256 → 128 → 64)")
print(f"✓ SpatialDropout1D for sequential data")
print(f"✓ Extended training patience: 8 epochs")
print(f"✓ Batch size: 32 (larger for stability)")
print(f"✓ Learning rate: 0.001 with scheduling")
print(f"✓ Lighter L2 regularization for better convergence")
print(f"✓ Class weights applied")
print("="*80)