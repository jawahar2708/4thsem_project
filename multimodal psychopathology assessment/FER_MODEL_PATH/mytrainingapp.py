import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


model = load_model('/my_fer_model1.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]            
        roi_gray = cv2.resize(roi_gray, (48, 48)) 
        img_pixels = roi_gray.astype('float32') / 255.0 
        img_pixels = np.expand_dims(img_pixels, axis=0) 
        img_pixels = np.expand_dims(img_pixels, axis=-1) 

        prediction = model.predict(img_pixels)
        max_index = np.argmax(prediction[0])
        predicted_emotion = emotion_labels[max_index]

        cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Facial Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()