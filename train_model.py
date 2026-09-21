import os
import json
import psycopg2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import uuid
from datetime import datetime

# --- Configuration ---
DB_CREDENTIALS = {
    'dbname': 'vit_ind_03_db',
    'user': 'postgres',
    'password': 'password',
    'host': 'localhost',
    'port': 5432
}

# The target time-width for our spectrograms. 
# MIMII audio files are typically 10 seconds long. We will pad/truncate to this width.
TARGET_TIME_STEPS = 400 

def get_available_machine_types():
    print("🔍 Checking available machine types in the database...")
    conn = psycopg2.connect(**DB_CREDENTIALS)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT Feature_Vector->>'machine_type' 
        FROM TBL_SIGNAL_FEATURE_STORE 
        WHERE Feature_Type = 'Spectrogram' AND Feature_Vector->>'machine_type' IS NOT NULL
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [row[0] for row in rows]

def fetch_data_from_db(machine_type):
    print(f"📥 Fetching Spectrograms for Machine Type: {machine_type}...")
    conn = psycopg2.connect(**DB_CREDENTIALS)
    cursor = conn.cursor()
    
    # Fetch Spectrograms only for the specified machine_type
    cursor.execute("""
        SELECT Feature_Vector FROM TBL_SIGNAL_FEATURE_STORE 
        WHERE Feature_Type = 'Spectrogram' 
        AND Feature_Vector->>'machine_type' = %s
    """, (machine_type,))
    rows = cursor.fetchall()
    
    X = []
    y = []
    
    print(f"🔄 Parsing {len(rows)} JSON payloads into NumPy arrays...")
    for row in rows:
        payload = row[0]
        label = payload['label']
        data_matrix = np.array(payload['data']) # Shape: (128, time_steps)
        
        # Standardize the time axis (width) to exactly TARGET_TIME_STEPS
        current_time_steps = data_matrix.shape[1]
        
        if current_time_steps < TARGET_TIME_STEPS:
            # Pad with zeros (silence) if too short
            pad_width = TARGET_TIME_STEPS - current_time_steps
            data_matrix = np.pad(data_matrix, ((0, 0), (0, pad_width)), mode='constant')
        elif current_time_steps > TARGET_TIME_STEPS:
            # Truncate if too long
            data_matrix = data_matrix[:, :TARGET_TIME_STEPS]
            
        X.append(data_matrix)
        y.append(label)
        
    cursor.close()
    conn.close()
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for CNN input: (Samples, Height, Width, Channels)
    if len(X) > 0:
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    return X, y

def build_cnn_model(input_shape):
    """Builds a lightweight CNN optimized for an Intel i9 CPU."""
    model = models.Sequential()
    
    # 1st Convolutional Block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # 2nd Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten the 2D maps to 1D vectors
    model.add(layers.Flatten())
    
    # Fully Connected Layer
    model.add(layers.Dense(64, activation='relu'))
    
    # Output Layer (Binary Classification: 0=Healthy, 1=Defective)
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def log_metrics_to_db(model_name, dataset_size, acc, prec, rec, f1):
    """Saves the model execution statistics back to our factory database."""
    print("💾 Logging KPI metrics to TBL_MODEL_TRAINING_HISTORY...")
    conn = psycopg2.connect(**DB_CREDENTIALS)
    cursor = conn.cursor()
    
    # 1. Register the Model
    model_id = str(uuid.uuid4())
    cursor.execute("""
        INSERT INTO TBL_AI_MODEL_MASTER 
        (Model_ID, Model_Name, Algorithm_Type, Framework, Version, Deployment_Type) 
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (model_id, model_name, 'CNN', 'TensorFlow/Keras', '1.0', 'Edge AI Ready'))
    
    # 2. Log the Training Run
    training_id = str(uuid.uuid4())
    cursor.execute("""
        INSERT INTO TBL_MODEL_TRAINING_HISTORY 
        (Training_ID, Model_ID, Dataset_Size, Training_Date, Accuracy, Precision, Recall, F1_Score)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (training_id, model_id, dataset_size, datetime.now().date(), acc, prec, rec, f1))
    
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    machine_types = get_available_machine_types()
    
    if not machine_types:
        print("❌ No machine types found in the database. Run bulk_feature_extractor.py first!")
        exit(1)
        
    print(f"✅ Found machine types to train: {machine_types}")
    
    for machine_type in machine_types:
        print(f"\n{'='*50}\nStarting Training Pipeline for: {machine_type.upper()}\n{'='*50}")
        
        # 1. Extract Data
        X, y = fetch_data_from_db(machine_type)
        if len(y) == 0:
            print(f"⚠️ No data found for {machine_type}. Skipping...")
            continue
            
        print(f"✅ Data Ready! Shape: {X.shape}, Total Samples: {len(y)}")
        
        # 2. Split Data (70% Train, 30% Test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 3. Build Model
        input_shape = (X.shape[1], X.shape[2], 1)
        model = build_cnn_model(input_shape)
        
        # 4. Train Model
        print(f"🚀 Starting AI Training for {machine_type}...")
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
        
        # 5. Blind Test (Reality Check)
        print("🧪 Running blind test on unseen data...")
        predictions_prob = model.predict(X_test)
        predictions = (predictions_prob > 0.5).astype(int).flatten()
        
        acc = float(accuracy_score(y_test, predictions))
        prec = float(precision_score(y_test, predictions, zero_division=0))
        rec = float(recall_score(y_test, predictions, zero_division=0))
        f1 = float(f1_score(y_test, predictions, zero_division=0))
        
        print("\n--- 🎯 MODEL RESULTS ---")
        print(f"Accuracy:  {acc*100:.2f}%")
        print(f"Precision: {prec*100:.2f}%")
        print(f"Recall:    {rec*100:.2f}%")
        print(f"F1-Score:  {f1*100:.2f}%")
        print("------------------------\n")
        
        # 6. Log and Save
        model_display_name = f"Acoustic Defect CNN - {machine_type}"
        log_metrics_to_db(model_display_name, len(y), acc, prec, rec, f1)
        
        model_filename = f"acoustic_cnn_model_{machine_type.lower()}.h5"
        model.save(model_filename)
        print(f"📦 Model fully trained and saved locally to {model_filename}!")
