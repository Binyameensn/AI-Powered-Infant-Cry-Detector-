# AI-Powered-Infant-Cry-Detector
A deep learning application that classifies the reason for a baby's cry (hunger, pain, etc.) from live or uploaded audio. Built with a TensorFlow/Keras CNN, Librosa for audio processing, and a responsive Flask web UI with real-time recording and visualization. Helps caregivers understand an infant's needs instantly.
<img width="1904" height="892" alt="Screenshot 2025-08-01 170936" src="https://github.com/user-attachments/assets/ecf0837f-e990-47ad-bc07-319f91505814" />
<img width="1708" height="867" alt="Screenshot 2025-08-01 171318" src="https://github.com/user-attachments/assets/8f760555-4e1d-45aa-a3a1-f6610359b7e4" />
<img width="1827" height="882" alt="Screenshot 2025-08-01 171028" src="https://github.com/user-attachments/assets/0ee76e09-d576-43f7-aa85-68f4a30e1274" />

This project is an end-to-end deep learning application designed to help parents and caregivers understand an infant's needs by analyzing the sound of their cry. Using a modern, responsive web interface, the system classifies a cry into one of several distinct categories—such as hunger, pain, discomfort, or tiredness—providing quick, data-driven insights to reduce guesswork and stress.

Key Features
Real-time Audio Recording: Capture and analyze cries directly in the browser using the MediaRecorder API.

File Upload Analysis: Analyze pre-recorded audio files (.wav, .mp3).

Live Audio Visualization: Displays a dynamic frequency graph while recording, providing instant visual feedback.

Accurate Deep Learning Model: Utilizes a Convolutional Neural Network (CNN) to classify cry reasons with high accuracy.

Modern & Responsive UI: A clean, mobile-first interface built with Flask and Tailwind CSS, ensuring a seamless experience on any device.

How It Works & Technology Stack
The application follows a classic machine learning pipeline. Audio data, whether uploaded or recorded live, is first processed on the backend using Pydub and Librosa. This workflow converts the raw audio into a Mel-spectrogram—a 2D visual representation of the sound.

These spectrograms are then fed into a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model, pre-trained on a large, labeled dataset of infant cries, analyzes the image and predicts the most likely reason.

The entire system is served by a lightweight but powerful Flask backend.

Backend: Python, Flask, TensorFlow/Keras

Audio Processing: Librosa, Pydub, FFmpeg

Frontend: HTML, Tailwind CSS, Modern JavaScript

Training Environment: Google Colab, Jupyter

Getting Started
The project is structured for easy setup. The data preparation and model training steps are detailed in their respective folders, with the final web application located in the 5_flask_app directory. Please see the setup instructions within each folder to run the project locally.
