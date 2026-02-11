import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sounddevice as sd
import librosa
import threading
import queue
import time


FER_MODEL_PATH = '/my_fer_model.h5'
SER_MODEL_PATH = '/my_audio_emotion_model.h5' 
SER_CLASSES_PATH = '/emotion_classes.npy'     

SAMPLE_RATE = 22050
DURATION = 2.5
SILENCE_THRESHOLD = 0.02


COMMON_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


current_audio_probs = np.ones(7) / 7.0 
audio_status = "Listening..."



def map_ser_to_fer(ser_probs, ser_classes):
   
    new_probs = np.zeros(7)
    
    for i, label in enumerate(ser_classes):
        prob = ser_probs[i]
        label = label.lower()
        
        if label == 'angry': new_probs[0] += prob
        elif label == 'disgust': new_probs[1] += prob
        elif label == 'fearful': new_probs[2] += prob
        elif label == 'happy': new_probs[3] += prob
        elif label == 'neutral': new_probs[4] += prob
        elif label == 'calm': new_probs[4] += prob 
        elif label == 'sad': new_probs[5] += prob
        elif label == 'surprised': new_probs[6] += prob
        
    return new_probs / np.sum(new_probs)

def audio_processing_thread():
   
    global current_audio_probs, audio_status
    
    print("Loading SER Model...")
    ser_model = load_model(SER_MODEL_PATH)
    ser_classes = np.load(SER_CLASSES_PATH)
    
    while True:
        try:
            
            recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            audio_data = recording.squeeze()
            
            
            rms = np.sqrt(np.mean(audio_data**2))
            if rms < SILENCE_THRESHOLD:
                audio_status = "Silence"
                
                continue 

            audio_status = "Processing..."
            
            max_val = np.max(np.abs(audio_data))
            if max_val > 0: audio_data = audio_data / max_val
            
            mfccs = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=40)
            mfccs_processed = np.mean(mfccs.T, axis=0)
            mfccs_input = np.expand_dims(mfccs_processed, axis=0)
            mfccs_input = np.expand_dims(mfccs_input, axis=2)
            
            pred = ser_model.predict(mfccs_input, verbose=0)[0]
            
            mapped_probs = map_ser_to_fer(pred, ser_classes)
            
            current_audio_probs = mapped_probs
            
            top_idx = np.argmax(mapped_probs)
            audio_status = f"Voice: {COMMON_LABELS[top_idx]}"
            
        except Exception as e:
            print(f"Audio Thread Error: {e}")

def main():
    
    t = threading.Thread(target=audio_processing_thread)
    t.daemon = True 
    t.start()
    
    print("Loading FER Model...")
    fer_model = load_model(FER_MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(0) 
    
    W_VIDEO = 0.5
    W_AUDIO = 0.5 

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
          
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = roi_gray.astype('float32') / 255.0
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels = np.expand_dims(img_pixels, axis=-1)
            
            fer_preds = fer_model.predict(img_pixels, verbose=0)[0] 
            
            fused_preds = (fer_preds * W_VIDEO) + (current_audio_probs * W_AUDIO)
            
        
            final_idx = np.argmax(fused_preds)
            final_emotion = COMMON_LABELS[final_idx]
            confidence = fused_preds[final_idx]
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            label = f"{final_emotion} ({confidence*100:.1f}%)"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Audio: {audio_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            vid_idx = np.argmax(fer_preds)
            vid_emo = COMMON_LABELS[vid_idx]
            cv2.putText(frame, f"V: {vid_emo}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow('Multimodal Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()