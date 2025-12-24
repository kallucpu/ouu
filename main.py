import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math

st.title("Blink Detection with Face Mesh")

# Load custom sound
audio_file = "audio.mp3"

# Initialize MediaPipe Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

# Eye landmarks for EAR calculation
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EYE_THRESHOLD = 0.25

# Helper functions
def euclidean(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def eye_aspect_ratio(landmarks, eye):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye]
    return (euclidean(p2, p6) + euclidean(p3, p5)) / (2 * euclidean(p1, p4))

# Webcam input via Streamlit
webrtc = st.camera_input("Look into the camera")

if webrtc is not None:
    # Convert to OpenCV format
    frame = np.array(webrtc)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Process frame with MediaPipe
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Draw landmarks
        for idx in LEFT_EYE + RIGHT_EYE:
            x = int(landmarks[idx].x * frame.shape[1])
            y = int(landmarks[idx].y * frame.shape[0])
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        # Calculate EAR
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2

        # Play alert if eyes closed
        if ear < EYE_THRESHOLD:
            st.audio(audio_file, start_time=0)

    # Display frame in browser
    st.image(frame, channels="BGR")

