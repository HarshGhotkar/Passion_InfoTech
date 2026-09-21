import os
import random
import time
import threading
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
import tensorflow as tf
import librosa
import numpy as np
import scipy.signal

import glob

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
socketio = SocketIO(app, cors_allowed_origins="*")

# Load all trained models globally when the server starts
print("Loading AI Models into Memory...")
models_dict = {}
try:
    model_files = glob.glob('acoustic_cnn_model_*.h5')
    # Fallback to the old generic model if no new ones exist yet
    if not model_files and os.path.exists('acoustic_cnn_model.h5'):
        model_files = ['acoustic_cnn_model.h5']
        
    for m_file in model_files:
        try:
            loaded_model = tf.keras.models.load_model(m_file)
            # Extract machine type from filename (e.g. acoustic_cnn_model_valve.h5 -> valve)
            if m_file == 'acoustic_cnn_model.h5':
                m_type = 'default'
            else:
                m_type = m_file.replace('acoustic_cnn_model_', '').replace('.h5', '').lower()
            models_dict[m_type] = loaded_model
            print(f"Model loaded for type '{m_type}': {m_file}")
        except Exception as e:
            print(f"Error loading {m_file}: {e}")
            
    if not models_dict:
        print("Warning: No models were successfully loaded!")
except Exception as e:
    print(f"Error searching for models: {e}")

import threading

background_thread = None
thread_lock = threading.Lock()

# --- Background Factory Simulator ---
def factory_simulation_thread():
    """Simulates real-time acoustic IoT sensor data and AI predictions streaming in."""
    while True:
        # We simulate the AI analyzing acoustic data from 5 different machines
        payload = {
            "karma_score": random.randint(90, 99),
            "machines": [
                {
                    "name": "CNC Machine",
                    "health": random.randint(85, 99),
                    "fail_prob": random.randint(1, 15),
                    "rec": "Healthy",
                    "status": "Safe"
                },
                {
                    "name": "Robotic Arm",
                    "health": random.randint(88, 99),
                    "fail_prob": random.randint(1, 12),
                    "rec": "Healthy",
                    "status": "Safe"
                },
                {
                    "name": "Gear Assembly",
                    "health": random.randint(55, 75),
                    "fail_prob": random.randint(25, 45),
                    "rec": "Schedule Maintenance",
                    "status": "Warning"
                },
                {
                    "name": "Metal Furnace",
                    "health": random.randint(90, 99),
                    "fail_prob": random.randint(1, 10),
                    "rec": "Healthy",
                    "status": "Safe"
                },
                {
                    "name": "Bearing",
                    "health": random.randint(15, 35),
                    "fail_prob": random.randint(65, 85),
                    "rec": "Immediate Shutdown",
                    "status": "Critical"
                }
            ]
        }
        socketio.emit('factory_update', payload)
        socketio.sleep(3) # Use socketio.sleep instead of time.sleep to prevent freezing!

@socketio.on('connect')
def handle_connect():
    global background_thread
    with thread_lock:
        if background_thread is None:
            background_thread = socketio.start_background_task(factory_simulation_thread)
    print("Live Dashboard Connected!")


# --- Audio Processing Logic ---
TARGET_TIME_STEPS = 400

def bandpass_filter(data, sr, lowcut=100.0, highcut=5000.0, order=5):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return scipy.signal.lfilter(b, a, data)

def process_audio(file_path):
    """Loads audio, extracts spectrogram, and formats it for the CNN."""
    y, sr = librosa.load(file_path, sr=None)
    y_filtered = bandpass_filter(y, sr)
    
    spectrogram = librosa.feature.melspectrogram(y=y_filtered, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    
    current_time_steps = spectrogram_db.shape[1]
    if current_time_steps < TARGET_TIME_STEPS:
        pad_width = TARGET_TIME_STEPS - current_time_steps
        spectrogram_db = np.pad(spectrogram_db, ((0, 0), (0, pad_width)), mode='constant')
    elif current_time_steps > TARGET_TIME_STEPS:
        spectrogram_db = spectrogram_db[:, :TARGET_TIME_STEPS]
        
    X = spectrogram_db.reshape(1, spectrogram_db.shape[0], spectrogram_db.shape[1], 1)
    return X

# --- API Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not models_dict:
        return jsonify({'error': 'No AI Models are loaded. Check server logs.'}), 500
        
    machine_type = request.form.get('machine_type', '').lower()
    
    if not machine_type:
        # If no specific type provided, try to use 'default' or just pick the first one
        if 'default' in models_dict:
            machine_type = 'default'
        else:
            return jsonify({'error': 'machine_type parameter is required (e.g. valve, pump).'}), 400
            
    if machine_type not in models_dict:
        available = list(models_dict.keys())
        return jsonify({'error': f"Model for '{machine_type}' not found. Available models: {available}"}), 404
        
    selected_model = models_dict[machine_type]
        
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded.'}), 400
        
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No audio file selected.'}), 400
        
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        try:
            X = process_audio(filepath)
            prediction_prob = selected_model.predict(X)[0][0]
            is_defective = prediction_prob > 0.5
            status = 'Defective (Crack/Fatigue)' if is_defective else 'Healthy'
            
            return jsonify({
                'status': 'success',
                'prediction': status,
                'probability_defective': float(prediction_prob),
                'probability_healthy': float(1.0 - prediction_prob)
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # Use socketio.run instead of app.run
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
