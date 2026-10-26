import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, TimeDistributed, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


print("Veri yükleniyor...")
df = pd.read_csv('Time_domain_subsamples.csv', header=None)

X_raw = df.iloc[:, 0:1800].values
y_raw = df.iloc[:, 1800].values

class_names = [
    'Stand', 'Sit', 'Talk-sit', 'Talk-stand', 'Stand-sit', 
    'Lay', 'Lay-stand', 'Pick', 'Jump', 'Push-up', 
    'Sit-up', 'Walk', 'Walk-backward', 'Walk-circle', 
    'Run', 'Stair-up', 'Stair-down', 'Table-tennis'
]

n_samples = X_raw.shape[0]
n_steps = 300
n_channels = 6 

X_reshaped = np.zeros((n_samples, n_steps, n_channels))

X_reshaped[:, :, 0] = X_raw[:, 0:300]      # Acc X
X_reshaped[:, :, 1] = X_raw[:, 300:600]    # Acc Y
X_reshaped[:, :, 2] = X_raw[:, 600:900]    # Acc Z
X_reshaped[:, :, 3] = X_raw[:, 900:1200]   # Gyro X
X_reshaped[:, :, 4] = X_raw[:, 1200:1500]  # Gyro Y
X_reshaped[:, :, 5] = X_raw[:, 1500:1800]  # Gyro Z

print(f"Veri yeniden şekillendirildi: {X_reshaped.shape}")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_raw)
y_categorical = to_categorical(y_encoded)

# Train-Test Split (%70 Train, %30 Test)
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.3, random_state=42, stratify=y_encoded)

n_subsequences = 4
sub_sequence_length = 75 # 300 / 4

X_train_cnn_lstm = X_train.reshape((X_train.shape[0], n_subsequences, sub_sequence_length, n_channels))
X_test_cnn_lstm = X_test.reshape((X_test.shape[0], n_subsequences, sub_sequence_length, n_channels))

print(f"Model girişi için şekil (Samples, Subseq, Steps, Channels): {X_train_cnn_lstm.shape}")

model = Sequential()

model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), 
                          input_shape=(n_subsequences, sub_sequence_length, n_channels)))

model.add(TimeDistributed(MaxPooling1D(pool_size=2)))

model.add(TimeDistributed(Flatten()))

model.add(LSTM(128, activation='tanh')) 
model.add(Dropout(0.5)) 

model.add(Dense(18, activation='softmax')) # KU-HAR 18 sınıf içeriyor

optimizer = Adam(learning_rate=0.0025)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print(model.summary())

history = model.fit(
    X_train_cnn_lstm, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_test_cnn_lstm, y_test),
    verbose=1
)

y_pred = model.predict(X_test_cnn_lstm)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Metrikler
acc = accuracy_score(y_true, y_pred_classes)
prec = precision_score(y_true, y_pred_classes, average='macro')
rec = recall_score(y_true, y_pred_classes, average='macro')
f1 = f1_score(y_true, y_pred_classes, average='macro')

print("\n--- Model Performans Metrikleri ---")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\n--- Sınıflandırma Raporu ---")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

plt.figure(figsize=(14, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Grafik 2: Confusion Matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

class_accuracies = cm.diagonal() / cm.sum(axis=1)

plt.figure(figsize=(14, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(class_names)))
bars = plt.bar(class_names, class_accuracies, color=colors)

plt.title('Accuracy rate for each activity (Recall)', fontsize=15)
plt.xlabel('Activities', fontsize=12)
plt.ylabel('Accuracy (0-1 Arası)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.1) 
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"%{yval*100:.1f}", 
             ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()