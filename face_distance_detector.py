

import streamlit as st
import cv2
import numpy as np
from io import BytesIO
import time

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_model(url):
    if url.startswith('http'):
        # Load video stream from URL
        cap = cv2.VideoCapture(url)
    else:
        # Load video stream from local camera
        cap = cv2.VideoCapture(int(url))

    if not cap.isOpened():
        raise Exception(f"Error: Unable to open camera at {url}")

    return cap

def detect_faces(frame, initial_distance=60):
    distance = None  # Initialize distance outside the loop

    if frame is not None:  # Check if the frame is valid
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            distance = round(initial_distance * 200 / w, 2)  # Update distance when face is detected
            cv2.putText(frame, f'Distance: {distance} cm', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return frame, distance  # Return the frame and calculated distance
    else:
        return None, None

def main():
    st.title('Live Face Distance Detection')

    camera_type = st.radio("Select Camera Type:", ('ESP32-CAM', 'Laptop/Desktop Camera'))
    if camera_type == 'ESP32-CAM':
        url = st.text_input('Enter ESP32-CAM video stream URL')
    else:
        url = st.text_input('Enter camera number (0 for default camera)')

    if not url:
        st.warning('Please enter the camera URL or number.')
        return

    cap = load_model(url)
    st.write('Press "Start Face Detection" to begin')

    start_detection = st.button('Start Face Detection')

    if start_detection:
        st.write('Face Detection started')
        while True:
            ret, frame = cap.read()  # Read frame from the camera

            if not ret:
                st.error('Error: Unable to read frame from the camera.')
                break

            # Perform face detection and get the frame with faces and the distance
            frame_with_faces, distance = detect_faces(frame)

            if frame_with_faces is not None:
                # Display the frame with face detection and distance measurements
                st.image(frame_with_faces, channels='BGR', use_column_width=True)

                # Display the live distance measurement
                if distance is not None:
                    st.write(f'Live Distance: {distance} cm')

            unique_key = f'stop_btn_{time.time()}'  # Generate a unique key using timestamp
            if st.button('Stop Detection', key=unique_key):
                st.write('Face Detection stopped')
                break

    cap.release()

if __name__ == '__main__':
    main()

