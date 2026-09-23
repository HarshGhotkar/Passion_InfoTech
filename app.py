import os
import random
import time
import threading
from flask import Flask, request, jsonify, render_template, url_for
from flask_socketio import SocketIO
import tensorflow as tf
import librosa
import librosa.display
import numpy as np
import scipy.signal
import glob
import uuid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

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

# Load Diagnostic KMeans Model
kmeans_model = None
try:
    if os.path.exists('kmeans_diagnostic.pkl'):
        with open('kmeans_diagnostic.pkl', 'rb') as f:
            kmeans_model = pickle.load(f)
        print("Diagnostic KMeans model loaded.")
    else:
        print("Diagnostic KMeans model not found.")
except Exception as e:
    print(f"Error loading KMeans model: {e}")

import threading

background_thread = None
thread_lock = threading.Lock()

total_tested = 0
healthy_count = 0
faulty_count = 0
diagnostic_counts = {
    "Normal Pattern Detection": 0,
    "Micro Crack Signature": 0,
    "Material Fatigue": 0,
    "Pressure/Seal Leak": 0,
    "Material Defect": 0
}

# --- Background Factory Simulator ---
def factory_simulation_thread():
    """Simulates real-time acoustic IoT sensor data and AI predictions streaming in."""
    global total_tested, healthy_count, faulty_count, diagnostic_counts
    
    # Get all available wav files in the dataset (6_dB_valve)
    dataset_files = glob.glob('6_dB_valve/**/*.wav', recursive=True)
    if not dataset_files:
        print("Warning: No .wav files found in 6_dB_valve directory for simulation.")
        
    while True:
        if dataset_files and ('valve' in models_dict or 'default' in models_dict):
            random_file = random.choice(dataset_files)
            try:
                X, img_url, mean_mfcc = process_audio(random_file)
                model_key = 'valve' if 'valve' in models_dict else 'default'
                selected_model = models_dict[model_key]
                prediction_prob = selected_model.predict(X)[0][0]
                is_defective = prediction_prob > 0.5
                
                total_tested += 1
                diagnostic_label = "Normal Pattern Detection"
                
                tag_id = uuid.uuid4().hex[:4].upper()
                
                if is_defective:
                    faulty_count += 1
                    part_tag = f"VLV-ERR-{tag_id}"
                    
                    if kmeans_model:
                        cluster_id = kmeans_model.predict([mean_mfcc])[0]
                        if cluster_id == 0:
                            diagnostic_label = "Micro Crack Signature"
                            rul = f"{random.randint(1, 4)} Hrs"
                        elif cluster_id == 1:
                            diagnostic_label = "Material Fatigue"
                            rul = f"{random.randint(300, 500)} Hrs"
                        else:
                            diagnostic_label = "Pressure/Seal Leak"
                            rul = f"{random.randint(48, 72)} Hrs"
                    else:
                        diagnostic_label = "Material Defect"
                        rul = "72 Hrs"
                        
                    status_text = "Critical" if "Crack" in diagnostic_label or "Leak" in diagnostic_label else "Warning"
                    rec_text = diagnostic_label
                    health_val = random.randint(15, 35)
                else:
                    healthy_count += 1
                    part_tag = f"VLV-OK-{tag_id}"
                    rul = ">5000 Hrs"
                    status_text = "Safe"
                    rec_text = "Healthy"
                    health_val = random.randint(85, 99)
                    
                # Increment specific diagnostic counter
                if diagnostic_label in diagnostic_counts:
                    diagnostic_counts[diagnostic_label] += 1
                else:
                    diagnostic_counts[diagnostic_label] = 1
                    
                karma = int((healthy_count / max(1, total_tested)) * 100)
                
                payload = {
                    "karma_score": karma,
                    "total_tested": total_tested,
                    "healthy_count": healthy_count,
                    "faulty_count": faulty_count,
                    "diagnostic_counts": diagnostic_counts,
                    "latest_spectrogram": img_url,
                    "diagnostic_log": {
                        "part_tag": part_tag,
                        "rul": rul,
                        "signal_source": "Live Valve Audio",
                        "analysis_type": "KMeans Feature Clustering" if is_defective else "CNN Classification",
                        "result": diagnostic_label,
                        "status": status_text
                    },
                    "machines": [
                        {
                            "name": "Live Valve Stream",
                            "health": health_val,
                            "fail_prob": int(prediction_prob * 100),
                            "rec": rec_text,
                            "status": status_text
                        }
                    ]
                }
                socketio.emit('factory_update', payload)
            except Exception as e:
                print(f"Simulation error processing {random_file}: {e}")
        else:
            payload = {
                "karma_score": 100,
                "total_tested": total_tested,
                "healthy_count": healthy_count,
                "faulty_count": faulty_count,
                "machines": []
            }
            socketio.emit('factory_update', payload)
            
        socketio.sleep(5) # Wait 5 seconds before testing the next file

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
    
    # Save spectrogram image
    img_name = f"spec_{uuid.uuid4().hex}.png"
    img_path = os.path.join('static', 'spectrograms', img_name)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()
    
    current_time_steps = spectrogram_db.shape[1]
    if current_time_steps < TARGET_TIME_STEPS:
        pad_width = TARGET_TIME_STEPS - current_time_steps
        spectrogram_db = np.pad(spectrogram_db, ((0, 0), (0, pad_width)), mode='constant')
    elif current_time_steps > TARGET_TIME_STEPS:
        spectrogram_db = spectrogram_db[:, :TARGET_TIME_STEPS]
        
    X = spectrogram_db.reshape(1, spectrogram_db.shape[0], spectrogram_db.shape[1], 1)
    
    # Extract mean MFCC for diagnostic clustering
    mfccs = librosa.feature.mfcc(y=y_filtered, sr=sr, n_mfcc=13)
    mean_mfcc = np.mean(mfccs, axis=1)
    
    return X, f"/static/spectrograms/{img_name}", mean_mfcc

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
            X, img_url, _ = process_audio(filepath)
            prediction_prob = selected_model.predict(X)[0][0]
            is_defective = prediction_prob > 0.5
            status = 'Defective (Crack/Fatigue)' if is_defective else 'Healthy'
            
            return jsonify({
                'status': 'success',
                'prediction': status,
                'probability_defective': float(prediction_prob),
                'probability_healthy': float(1.0 - prediction_prob),
                'spectrogram_url': img_url
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
