import glob
import librosa
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
    mfccs = librosa.feature.mfcc(y=y_filtered, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

def main():
    print("Loading KMeans model...")
    try:
        with open('kmeans_diagnostic.pkl', 'rb') as f:
            kmeans = pickle.load(f)
    except FileNotFoundError:
        print("Model not found. Please run train_diagnostic_cluster.py first.")
        return

    print("Fetching sample of abnormal audio files...")
    abnormal_files = glob.glob('6_dB_valve/**/abnormal/*.wav', recursive=True)
    
    if not abnormal_files:
        print("No abnormal files found.")
        return
        
    np.random.shuffle(abnormal_files)
    sample_files = abnormal_files[:200]  # Take 200 for a quick visual verification

    print(f"Extracting features for {len(sample_files)} files...")
    X_features = []
    for idx, f in enumerate(sample_files):
        X_features.append(extract_mean_mfcc(f))
        if (idx+1) % 50 == 0:
            print(f"Processed {idx+1} files...")
            
    X_features = np.array(X_features)
    
    print("Predicting clusters...")
    clusters = kmeans.predict(X_features)
    
    print("Applying PCA for dimensionality reduction (13D -> 2D)...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_features)
    
    print("Generating scatter plot...")
    plt.figure(figsize=(10, 7))
    
    # Map colors to our diagnostic labels
    colors = ['red', 'orange', 'blue']
    labels = ['Cluster 0: Micro Crack Signature', 'Cluster 1: Material Fatigue', 'Cluster 2: Pressure/Seal Leak']
    
    for i in range(3):
        # Select data points belonging to cluster i
        idx = np.where(clusters == i)
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], c=colors[i], label=labels[i], alpha=0.7, edgecolors='k', s=70)
        
    plt.title('Diagnostic AI: KMeans Cluster Verification (PCA Projection)', fontsize=14, pad=15)
    plt.xlabel('Principal Component 1 (Primary Acoustic Variance)', fontsize=12)
    plt.ylabel('Principal Component 2 (Secondary Acoustic Variance)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_img = 'kmeans_verification_plot.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"\nVerification plot saved successfully as: {output_img}")
    
if __name__ == '__main__':
    main()
