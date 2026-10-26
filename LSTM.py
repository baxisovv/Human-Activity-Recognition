# ================================
# KU-HAR LSTM TRAINING PIPELINE
# ================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ================================
# 1. LOAD DATA
# ================================

CSV_PATH = "Time_domain_subsamples.csv"  # <-- GEREKİRSE GÜNCELLE

# Header yok
df = pd.read_csv(CSV_PATH, header=None)

# Eksik verileri temizle
df.dropna(inplace=True)

print("Dataset shape (after dropna):", df.shape)

# ================================
# 2. FEATURE & LABEL SEPARATION
# ================================

# Son 2 sütun ID, onları atıyoruz
X = df.iloc[:, :1800].values   # 0 - 1799
y = df.iloc[:, 1800].values   # class label (0-17)

print("Raw X shape:", X.shape)
print("Raw y shape:", y.shape)

# ================================
# 3. RESHAPE & TRANSPOSE (ÇOK KRİTİK)
# ================================

# Önce (N, 6, 300)
X = X.reshape(-1, 6, 300)

# Sonra (N, 300, 6)
X = X.transpose(0, 2, 1)

print("X shape after reshape & transpose:", X.shape)
# BEKLENEN: (N, 300, 6)

# ================================
# 4. SCALING (StandardScaler)
# ================================

# LSTM için 2D'e indir -> scale -> tekrar 3D
N, T, F = X.shape
X_2d = X.reshape(N * T, F)

scaler = StandardScaler()
X_2d = scaler.fit_transform(X_2d)

X = X_2d.reshape(N, T, F)

print("X shape after scaling:", X.shape)

# ================================
# 5. LABEL ENCODING (ONE-HOT)
# ================================

num_classes = 18
y_cat = to_categorical(y, num_classes=num_classes)

print("y one-hot shape:", y_cat.shape)

# ================================
# 6. TRAIN / TEST SPLIT
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ================================
# 7. MODEL DEFINITION (LSTM)
# ================================

model = Sequential()

model.add(Bidirectional(
    LSTM(64, return_sequences=True),
    input_shape=(300, 6)
))
model.add(Dropout(0.3))

model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dense(18, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ================================
# 8. CALLBACKS
# ================================

callbacks = [
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
]

# ================================
# 9. TRAIN MODEL
# ================================

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ================================
# 10. TRAINING CURVES
# ================================

plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# ================================
# 11. EVALUATION
# ================================

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

class_names = [
    'Stand', 'Sit', 'Talk-sit', 'Talk-stand', 'Stand-sit',
    'Lay', 'Lay-stand', 'Pick', 'Jump', 'Push-up',
    'Sit-up', 'Walk', 'Walk-backward', 'Circle-walk',
    'Run', 'Stair-up', 'Stair-down', 'Table-tennis'
]

print("\nCLASSIFICATION REPORT\n")
print(classification_report(
    y_true_classes,
    y_pred_classes,
    target_names=class_names
))

# ================================
# 12. CONFUSION MATRIX
# ================================

cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
