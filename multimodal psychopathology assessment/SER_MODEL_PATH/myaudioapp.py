import os
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = 'my_audio_emotion_model.h5'
CLASSES_PATH = 'emotion_classes.npy'
SAMPLE_RATE = 22050
DURATION = 2.5
N_MFCC = 40

SILENCE_THRESHOLD = 0.02 

model = load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH)
print(f"Loaded Classes: {classes}")

def predict_emotion(audio_data, sr):
   
    rms = np.sqrt(np.mean(audio_data**2))
    if rms < SILENCE_THRESHOLD:
        return "Silence", 0.0

    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
    
    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=N_MFCC)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        
        mfccs_input = np.expand_dims(mfccs_processed, axis=0)
        mfccs_input = np.expand_dims(mfccs_input, axis=2)
        
        prediction = model.predict(mfccs_input, verbose=0)
        max_index = np.argmax(prediction[0])
        predicted_label = classes[max_index]
        confidence = prediction[0][max_index]
        
        return predicted_label, confidence
    except Exception as e:
        print(f"Error: {e}")
        return "Error", 0.0


print("Listening... (Speak clearly!)")
print(f"Silence Threshold: {SILENCE_THRESHOLD}")


try:
    while True:
        
        recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        
        audio_data = recording.squeeze()
        
        
        emotion, conf = predict_emotion(audio_data, SAMPLE_RATE)
        
        
        if emotion == "Silence":
            print(".", end="", flush=True) 
        else:
            print(f"\nDETECTED: {emotion.upper()} ({conf*100:.1f}%)")

except KeyboardInterrupt:
    print("\nStopping...") 
