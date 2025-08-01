import os
import librosa
import numpy as np
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, '..', 'dataset')
OUTPUT_PATH = os.path.join(BASE_DIR, '..', 'model_training') 


SAMPLE_RATE = 22050
DURATION = 5  
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MELS = 128 
HOP_LENGTH = 512
N_FFT = 2048

def preprocess_dataset(dataset_path, output_path):
  
    
    data = {
        "mappings": [],
        "labels": [],
        "features": []
    }

    print("Processing dataset...")

    
    os.makedirs(output_path, exist_ok=True)

   
    for i, (dirpath, _, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            category = os.path.basename(dirpath)
            data["mappings"].append(category)
            print(f"\nProcessing category: {category}")

            for f in filenames:
                if f.endswith(('.wav', '.mp3')):
                    file_path = os.path.join(dirpath, f)
                    
                    try:
        
                        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                        if len(signal) > SAMPLES_PER_TRACK:
                            signal = signal[:SAMPLES_PER_TRACK] 
                        else:
                            signal = np.pad(signal, (0, SAMPLES_PER_TRACK - len(signal)), 'constant') 
                            
                        mel_spectrogram = librosa.feature.melspectrogram(
                            y=signal, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
                        )
                        
                
                        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
                        
                    
                        data["features"].append(log_mel_spectrogram.tolist()) 
                        data["labels"].append(i - 1) 
                        print(f"  Processed: {f}")

                    except Exception as e:
                        print(f"Could not process file {file_path}: {e}")


    output_filepath = os.path.join(output_path, "processed_data.json")
    with open(output_filepath, "w") as fp:
        json.dump(data, fp, indent=4)
        
    print(f"\nData processing complete. Processed data saved to {output_filepath}")


if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, OUTPUT_PATH)