import numpy as np
import cv2
import tensorflow as tf
import keras
import os
import cv2
import random
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load the face and eye cascade classifiers from OpenCV
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')  # Changed classifier

new_model = tf.keras.models.load_model('Drowsiness_Detection_SOFTMAX_Main.h5')

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

counter = 0
eyes_roi = None  # Initialize eyes_roi to None

while True:
    ret, frame = cap.read()
    if not ret:  
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]  # Face region in grayscale
        roi_color = frame[y:y+h, x:x+w]  # Face region in color
        
        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
        
        if len(eyes) == 2:  # Ensure both eyes are detected
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                eyes_roi = roi_color[ey:ey+eh, ex:ex+ew]  # Crop the eye region

    if eyes_roi is not None:  # Check if eyes_roi has been set
        final_image = cv2.resize(eyes_roi, (224, 224))
        final_image = np.expand_dims(final_image, axis=0)  # Add batch dimension
        final_image = final_image / 255.0

        Prediction = new_model.predict(final_image)
        print(Prediction)
        font = cv2.FONT_HERSHEY_SIMPLEX
        Predictions = max(Prediction[0])
        print(Predictions)
        
        if Predictions >= 0.98:
            status = "Open Eyes"
            cv2.putText(frame, status, (150, 150), font, 3, (0, 255, 0), 2, cv2.LINE_4)

            x1, y1, w1, h1 = 0, 0, 175, 75
            cv2.rectangle(frame, (x1, x1), (x1+w1, y1+h1), (0, 0, 0), -1)
            cv2.putText(frame, 'Active', (x1+int(w1/10), y1+int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            counter += 1
            status = "Closed Eyes"
            cv2.putText(frame, status, (150, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            if counter > 5:
                x1, y1, w1, h1 = 0, 0, 175, 75
                cv2.rectangle(frame, (x1, x1), (x1+w1, y1+h1), (0, 0, 0), -1)
                cv2.putText(frame, 'Sleep Alert !!', (x1+int(w1/10), y1+int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Play an alarm
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load("alarm-alert-sound-effect-230557.mp3")
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
                counter = 0

    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
