import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from joblib import dump
import matplotlib.pyplot as plt

# === Read CSV in chunks ===
chunk_size = 10000  # Adjust the chunk size based on your available memory
chunks = pd.read_csv(r"C:\Users\3com\OneDrive\Desktop\hand project\Ninapro_DB1.csv", chunksize=chunk_size)

# Combine all chunks into a single DataFrame
df = pd.concat(chunks, ignore_index=True)

# === Filter for relevant gestures (labels 1-5, 11, 12) ===
selected_labels = [1, 2, 3, 4, 5, 11, 12]
df = df[df['restimulus'].isin(selected_labels)]

# === Fill NaNs just in case ===
df = df.ffill()

# === Normalize EMG channels ===
emg_data = df[[f'emg_{i}' for i in range(10)]].values
scaler = StandardScaler()
emg_data = scaler.fit_transform(emg_data)

# === Label vector ===
labels = df['restimulus'].values

# === Segment the data (sliding window) ===
def create_windows(data, labels, window_size=200, step=100):
    X, y = [], []
    for start in range(0, len(data) - window_size, step):
        end = start + window_size
        label_window = labels[start:end]
        if np.all(label_window == label_window[0]):  # consistent label
            X.append(data[start:end])
            y.append(label_window[0])
    return np.array(X), np.array(y)

X_windows, y_windows = create_windows(emg_data, labels)

# === Feature extraction: mean absolute value (MAV) for each channel ===
X_features = np.mean(np.abs(X_windows), axis=1)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X_features, y_windows, test_size=0.2, random_state=42)

# === Train classifier ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === Save model and scaler for later use (Arduino integration, etc.) ===
dump(clf, "emg_model.joblib")
dump(scaler, "emg_scaler.joblib")

# # === Optional: plot confusion matrix ===
# import seaborn as sns
# plt.figure(figsize=(8,6))
# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
#             xticklabels=selected_labels, yticklabels=selected_labels)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Gesture Confusion Matrix')
# plt.show()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from joblib import dump
import matplotlib.pyplot as plt

# === Load and combine CSV chunks ===
chunks = pd.read_csv(r"C:\Users\3com\OneDrive\Desktop\hand project\Ninapro_DB1.csv", chunksize=10000)
df = pd.concat(chunks, ignore_index=True)

# === Filter relevant gestures ===
selected_labels = [1, 2, 3, 4, 5, 11, 12]
df = df[df['restimulus'].isin(selected_labels)]
df = df.ffill()

# === Normalize EMG data ===
emg_data = df[[f'emg_{i}' for i in range(10)]].values
scaler = StandardScaler()
emg_data = scaler.fit_transform(emg_data)
labels = df['restimulus'].values

# === Sliding window function ===
def create_windows(data, labels, window_size=200, step=100):
    X, y = [], []
    for start in range(0, len(data) - window_size, step):
        end = start + window_size
        label_window = labels[start:end]
        if np.all(label_window == label_window[0]):
            X.append(data[start:end])
            y.append(label_window[0])
    return np.array(X), np.array(y)

X_windows, y_windows = create_windows(emg_data, labels)

# === Feature extraction: add multiple time-domain features ===
def extract_features(windows):
    features = []
    for window in windows:
        mav = np.mean(np.abs(window), axis=0)              # Mean Absolute Value
        ssc = np.sum(np.diff(window, axis=0) > 0, axis=0)  # Slope Sign Changes
        wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)  # Waveform Length
        rms = np.sqrt(np.mean(window**2, axis=0))          # Root Mean Square
        combined = np.hstack([mav, ssc, wl, rms])
        features.append(combined)
    return np.array(features)

X_features = extract_features(X_windows)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_windows, test_size=0.2, random_state=42, stratify=y_windows
)

# === Train Improved Random Forest ===
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_train, y_train)

# === Evaluate Model ===
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === Save model and scaler ===
dump(clf, "emg_model_improved.joblib")
dump(scaler, "emg_scaler_improved.joblib")

# === Optional: plot confusion matrix ===
# import seaborn as sns
# plt.figure(figsize=(8,6))
# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
#             xticklabels=selected_labels, yticklabels=selected_labels)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Gesture Confusion Matrix')
# plt.show()
