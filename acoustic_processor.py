import librosa
import numpy as np
import scipy.signal
import psycopg2
import json
import uuid

def bandpass_filter(data, sr, lowcut=100.0, highcut=5000.0, order=5):
    """
    Apply a Butterworth bandpass filter to the audio data for basic noise filtering.
    Frequencies outside [lowcut, highcut] are attenuated.
    """
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    
    # Generate Butterworth filter coefficients
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    
    # Apply filter to the data
    filtered_data = scipy.signal.lfilter(b, a, data)
    return filtered_data

def extract_features(file_path):
    """
    Load raw acoustic recording, apply noise filtering, and extract MFCC and Spectrogram features.
    """
    print(f"Loading raw audio file: {file_path}...")
    # 1. Load raw acoustic recording (.wav format)
    # sr=None preserves the original sampling rate of the file
    y, sr = librosa.load(file_path, sr=None)
    
    print(f"Applying digital signal processing (bandpass filter)...")
    # 2. Apply digital signal processing for basic noise filtering
    y_filtered = bandpass_filter(y, sr)
    
    print("Extracting MFCC and Spectrogram features...")
    # 3. Extract features
    # Mel-frequency cepstral coefficients (MFCCs)
    mfcc = librosa.feature.mfcc(y=y_filtered, sr=sr, n_mfcc=13)
    
    # Mel-scaled Spectrogram (converted to decibel scale)
    spectrogram = librosa.feature.melspectrogram(y=y_filtered, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Convert numpy arrays to nested lists for JSON serialization
    mfcc_json = mfcc.tolist()
    spectrogram_json = spectrogram_db.tolist()
    
    return mfcc_json, spectrogram_json

def insert_features_to_db(audio_id, mfcc_features, spectrogram_features, db_params):
    """
    Connect to PostgreSQL and insert extracted feature vectors into TBL_SIGNAL_FEATURE_STORE.
    """
    conn = None
    try:
        # Establish the database connection
        print("Connecting to PostgreSQL database...")
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        
        insert_query = """
            INSERT INTO TBL_SIGNAL_FEATURE_STORE 
            (Audio_ID, Feature_Type, Feature_Vector, Extraction_Method) 
            VALUES (%s, %s, %s, %s)
        """
        
        print("Inserting MFCC features into TBL_SIGNAL_FEATURE_STORE...")
        cursor.execute(insert_query, (
            audio_id,
            'MFCC',
            json.dumps(mfcc_features),  # Serialize list to JSON payload
            'librosa.feature.mfcc (Bandpass Filtered)'
        ))
        
        print("Inserting Spectrogram features into TBL_SIGNAL_FEATURE_STORE...")
        cursor.execute(insert_query, (
            audio_id,
            'Spectrogram',
            json.dumps(spectrogram_features), # Serialize list to JSON payload
            'librosa.feature.melspectrogram (Bandpass Filtered)'
        ))
        
        # Commit the transaction
        conn.commit()
        cursor.close()
        print("Successfully inserted features into database.")
        
    except psycopg2.Error as e:
        print(f"Database error occurred: {e}")
        if conn:
            conn.rollback()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    # --- Configuration Section ---
    
    # Database connection parameters
    # Update these values to match your actual PostgreSQL environment
    DB_CREDENTIALS = {
        'dbname': 'vit_ind_03_db',
        'user': 'postgres',
        'password': 'your_password',
        'host': 'localhost',
        'port': 5432
    }
    
    # Sample file to process (Ensure you have a .wav file at this path)
    SAMPLE_WAV_FILE = "machine_sound_sample.wav"
    
    # Example Audio_ID (This UUID should correspond to an existing record in TBL_AUDIO_SIGNAL_DATA)
    # In production, this ID would be passed in after the raw file is registered in the DB
    EXAMPLE_AUDIO_ID = str(uuid.uuid4())
    
    try:
        # Step 1 & 2: Process Audio and Extract Features
        # Uncomment the lines below when you have a valid .wav file to test with
        
        # mfcc, spectrogram = extract_features(SAMPLE_WAV_FILE)
        
        # Step 3: Insert into Database
        # insert_features_to_db(EXAMPLE_AUDIO_ID, mfcc, spectrogram, DB_CREDENTIALS)
        
        print("Script execution completed. (Ensure you uncomment the function calls and provide a valid .wav file to run the full pipeline.)")
        
    except FileNotFoundError:
        print(f"Error: The audio file '{SAMPLE_WAV_FILE}' was not found. Please provide a valid .wav file.")
    except Exception as e:
        print(f"Process failed: {e}")
