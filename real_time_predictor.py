import serial
import time
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# === Load the trained model and scaler ===
model = joblib.load("emg_model_improved.joblib")
scaler = joblib.load("emg_scaler_improved.joblib")

# === Set up serial connection to Arduino ===
arduino = serial.Serial('COM3', 9600)  # Replace COM3 with your correct port
time.sleep(2)  # Give Arduino time to reset

# === Simulate EMG signal (replace with real Arduino readings) ===
def analog_value():
    return np.random.randint(300, 700)  # Replace with serial reading in real setup

# === Feature extraction function: must match training code ===
def extract_features(emg_window):
    mav = np.mean(np.abs(emg_window), axis=0)               # Mean Absolute Value
    ssc = np.sum(np.diff(emg_window, axis=0) > 0, axis=0)   # Slope Sign Changes
    wl = np.sum(np.abs(np.diff(emg_window, axis=0)), axis=0)  # Waveform Length
    rms = np.sqrt(np.mean(emg_window ** 2, axis=0))         # Root Mean Square
    combined = np.hstack([mav, ssc, wl, rms])
    return combined.reshape(1, -1)

# === Predict gesture from EMG window ===
def predict_gesture(emg_window):
    features = extract_features(emg_window)
    scaled_features = scaler.transform(features)
    gesture = model.predict(scaled_features)[0]
    return gesture

print("Real-time EMG gesture prediction started. Press Ctrl+C to stop.")

try:
    while True:
        window = []
        for _ in range(200):  # Window size = 200 samples
            signal = [analog_value() for _ in range(10)]  # Replace with actual EMG reading
            window.append(signal)
            time.sleep(0.005)  # ~200Hz sampling rate

        emg_window = np.array(window)
        gesture = predict_gesture(emg_window)

        print(f"Predicted gesture: {gesture}")
        arduino.write(f"{gesture}\n".encode())  # Send gesture ID to Arduino

except KeyboardInterrupt:
    print("Stopped by user.")
    arduino.close()
