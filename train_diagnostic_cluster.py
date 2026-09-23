import glob
import librosa
import numpy as np
import pickle
from sklearn.cluster import KMeans
import scipy.signal

def bandpass_filter(data, sr, lowcut=100.0, highcut=5000.0, order=5):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return scipy.signal.lfilter(b, a, data)

def extract_mean_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y_filtered = bandpass_filter(y, sr)
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y_filtered, sr=sr, n_mfcc=13)
    # Take the mean across the time axis to get a single 13D vector per audio file
    mean_mfcc = np.mean(mfccs, axis=1)
    return mean_mfcc

def main():
    print("Finding abnormal audio files...")
    # Get only abnormal files to cluster the defect types
    abnormal_files = glob.glob('6_dB_valve/**/abnormal/*.wav', recursive=True)
    
    if not abnormal_files:
        print("No abnormal files found. Make sure dataset is present.")
        return
        
    # To save time in training, limit to a maximum of 500 files
    if len(abnormal_files) > 500:
        np.random.shuffle(abnormal_files)
        abnormal_files = abnormal_files[:500]
        
    print(f"Found {len(abnormal_files)} abnormal files. Extracting MFCC features...")
    
    X_features = []
    for idx, f in enumerate(abnormal_files):
        if idx % 50 == 0:
            print(f"Processed {idx}/{len(abnormal_files)} files...")
        feat = extract_mean_mfcc(f)
        X_features.append(feat)
        
    X_features = np.array(X_features)
    print(f"Feature matrix shape: {X_features.shape}")
    
    print("Training KMeans Clustering Model (3 Clusters)...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    kmeans.fit(X_features)
    
    model_path = 'kmeans_diagnostic.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(kmeans, f)
        
    print(f"Diagnostic clustering model saved to {model_path}!")

if __name__ == '__main__':
    main()
