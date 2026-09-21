import os
import glob
import librosa
import numpy as np
import scipy.signal
import psycopg2
import json
import uuid
from datetime import datetime

# --- Configuration ---
# Update this path to where your extracted MIMII folders are located
# E.g. 'D:\\Passion_InfoTech' should contain '6_dB_valve', '6_dB_pump', etc.
DATASET_ROOT_DIR = r"D:\Passion_InfoTech" 

DB_CREDENTIALS = {
    'dbname': 'vit_ind_03_db',
    'user': 'postgres',
    'password': 'password',
    'host': 'localhost',
    'port': 5432
}

# --- DSP Functions (From acoustic_processor.py) ---
def bandpass_filter(data, sr, lowcut=100.0, highcut=5000.0, order=5):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return scipy.signal.lfilter(b, a, data)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y_filtered = bandpass_filter(y, sr)
    
    mfcc = librosa.feature.mfcc(y=y_filtered, sr=sr, n_mfcc=13)
    spectrogram = librosa.feature.melspectrogram(y=y_filtered, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    
    return mfcc.tolist(), spectrogram_db.tolist(), librosa.get_duration(y=y, sr=sr)

# --- Database Functions ---
def get_or_create_mock_sensor(cursor, machine_name, machine_type):
    """Creates or fetches a mock sensor for our training dataset."""
    # First, get or create the machine
    cursor.execute("SELECT Machine_ID FROM TBL_MACHINE_MASTER WHERE Machine_Type = %s", (machine_type,))
    machine_result = cursor.fetchone()
    
    if machine_result:
        machine_id = machine_result[0]
    else:
        machine_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO TBL_MACHINE_MASTER (Machine_ID, Machine_Name, Machine_Type) 
            VALUES (%s, %s, %s)
        """, (machine_id, machine_name, machine_type))
        
    # Now get or create a sensor for this machine
    cursor.execute("SELECT Sensor_ID FROM TBL_SENSOR_MASTER WHERE Machine_ID = %s", (machine_id,))
    sensor_result = cursor.fetchone()
    
    if sensor_result:
        return sensor_result[0]
        
    sensor_id = str(uuid.uuid4())
    cursor.execute("""
        INSERT INTO TBL_SENSOR_MASTER (Sensor_ID, Machine_ID, Sensor_Type, Status) 
        VALUES (%s, %s, %s, %s)
    """, (sensor_id, machine_id, 'Acoustic Sensor', 'Active'))
    
    return sensor_id

def patch_legacy_data(cursor):
    """Updates older feature vectors in the DB that don't have the machine_type in their JSON."""
    print("🛠️ Checking for legacy data that needs patching...")
    cursor.execute("""
        UPDATE TBL_SIGNAL_FEATURE_STORE
        SET Feature_Vector = jsonb_set(Feature_Vector, '{machine_type}', '"Valve"', true)
        WHERE Feature_Vector->>'machine_type' IS NULL;
    """)
    # We don't commit here, we'll commit it in the main function

def process_dataset():
    conn = None
    try:
        conn = psycopg2.connect(**DB_CREDENTIALS)
        cursor = conn.cursor()
        
        patch_legacy_data(cursor)
        conn.commit()
        
        # Identify dataset directories
        machine_dirs = [d for d in glob.glob(os.path.join(DATASET_ROOT_DIR, "6_dB_*")) if os.path.isdir(d)]
        
        if not machine_dirs:
            print(f"❌ No dataset folders found in {DATASET_ROOT_DIR}. Please check the path!")
            return
            
        for dataset_dir in machine_dirs:
            machine_type_raw = os.path.basename(dataset_dir).replace('6_dB_', '') # e.g., 'valve', 'pump'
            machine_type = machine_type_raw.capitalize()
            
            # Check if we already have data for this machine type to avoid duplicate processing
            cursor.execute("""
                SELECT COUNT(*) FROM TBL_SIGNAL_FEATURE_STORE 
                WHERE Feature_Vector->>'machine_type' = %s
            """, (machine_type,))
            count = cursor.fetchone()[0]
            
            if count > 0:
                print(f"⏩ Skipping '{machine_type}'... Already found {count} records in the database.")
                continue
                
            machine_name = f"MIMII {machine_type} Training Rig"
            
            sensor_id = get_or_create_mock_sensor(cursor, machine_name, machine_type)
            
            # Look for .wav files recursively
            search_pattern = os.path.join(dataset_dir, "**", "*.wav")
            wav_files = glob.glob(search_pattern, recursive=True)
            
            if not wav_files:
                print(f"⚠️ No .wav files found in {dataset_dir}. Skipping...")
                continue
                
            print(f"✅ Found {len(wav_files)} audio files for {machine_type}. Starting bulk extraction...")
            
            success_count = 0
            for idx, file_path in enumerate(wav_files):
                try:
                    # 1. Determine ground truth label from folder name
                    label = 'abnormal' if 'abnormal' in file_path.lower() else 'normal'
                    is_defective = 1 if label == 'abnormal' else 0
                    
                    # 2. Extract features
                    mfcc, spectrogram, duration = extract_features(file_path)
                    
                    # 3. Insert into TBL_AUDIO_SIGNAL_DATA
                    audio_id = str(uuid.uuid4())
                    cursor.execute("""
                        INSERT INTO TBL_AUDIO_SIGNAL_DATA 
                        (Audio_ID, Sensor_ID, File_Path, Duration_Seconds) 
                        VALUES (%s, %s, %s, %s)
                    """, (audio_id, sensor_id, file_path, duration))
                    
                    # 4. Insert into TBL_SIGNAL_FEATURE_STORE
                    # We package the label and machine type inside the JSONB payload
                    mfcc_payload = json.dumps({"label": is_defective, "machine_type": machine_type, "data": mfcc})
                    spectrogram_payload = json.dumps({"label": is_defective, "machine_type": machine_type, "data": spectrogram})
                    
                    cursor.execute("""
                        INSERT INTO TBL_SIGNAL_FEATURE_STORE 
                        (Audio_ID, Feature_Type, Feature_Vector, Extraction_Method) 
                        VALUES (%s, %s, %s, %s)
                    """, (audio_id, 'MFCC', mfcc_payload, 'librosa.feature.mfcc (Bandpass)'))
                    
                    cursor.execute("""
                        INSERT INTO TBL_SIGNAL_FEATURE_STORE 
                        (Audio_ID, Feature_Type, Feature_Vector, Extraction_Method) 
                        VALUES (%s, %s, %s, %s)
                    """, (audio_id, 'Spectrogram', spectrogram_payload, 'librosa.feature.melspectrogram (Bandpass)'))
                    
                    # Commit every 50 files
                    if idx % 50 == 0:
                        conn.commit()
                        print(f"  Progress ({machine_type}): Processed {idx}/{len(wav_files)} files...")
                        
                    success_count += 1
                    
                except Exception as file_e:
                    print(f"Error processing file {file_path}: {file_e}")
            
            # Commit after each machine type
            conn.commit()
            print(f"🎉 Completed Extraction for {machine_type}! Successfully processed {success_count}/{len(wav_files)} files.\n")
            
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    process_dataset()
