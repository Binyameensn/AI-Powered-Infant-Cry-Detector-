import os
import librosa
import numpy as np
import tensorflow as tf
import json


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'saved_model', 'cry_detection_model.h5')

MAPPINGS_PATH = os.path.join(BASE_DIR, '..', 'model_training', 'processed_data.json')


SAMPLE_RATE = 22050
DURATION = 5  # sec
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

class CryPredictor:
    """
    Singleton class to load the model and mappings once and provide predictions.
    """
    model = None
    mappings = None
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CryPredictor, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model = tf.keras.models.load_model(MODEL_PATH)
        with open(MAPPINGS_PATH, "r") as fp:
            data = json.load(fp)
            self.mappings = data["mappings"]
        self._initialized = True

    def predict(self, audio_file_path):
        """
        Predicts the cry reason for a given audio file.
        :param audio_file_path (str): Path to the audio file.
        :return (str, float): Predicted reason and confidence score.
        """
        try:
           
            signal, sr = librosa.load(audio_file_path, sr=SAMPLE_RATE)

            
            if len(signal) > SAMPLES_PER_TRACK:
                signal = signal[:SAMPLES_PER_TRACK]
            else:
                signal = np.pad(signal, (0, SAMPLES_PER_TRACK - len(signal)), 'constant')

           
            mel_spectrogram = librosa.feature.melspectrogram(
                y=signal, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
            )
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

         
            log_mel_spectrogram = log_mel_spectrogram[np.newaxis, ..., np.newaxis]

            predictions = self.model.predict(log_mel_spectrogram)
            predicted_index = np.argmax(predictions)
            
            predicted_reason = self.mappings[predicted_index]
            confidence = float(predictions[0][predicted_index])

            return predicted_reason, confidence

        except Exception as e:
            return f"Error processing file: {e}", 0.0

def main():
    """
    Main function to run the prediction.
    """
    
    predictor = CryPredictor()

    test_audio_path = os.path.join(BASE_DIR, '..', 'dataset', 'pain', 'pain_1.wav') 

    if not os.path.exists(test_audio_path):
        print(f"Error: Test audio file not found at {test_audio_path}")
        print("Please replace the 'test_audio_path' variable with a valid file path.")
        return

    reason, confidence = predictor.predict(test_audio_path)
    
    print("\n--- Prediction Result ---")
    print(f"File: {os.path.basename(test_audio_path)}")
    print(f"Predicted Reason: {reason}")
    print(f"Confidence: {confidence:.2%}")
    print("-------------------------\n")


if __name__ == "__main__":
    main()