import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.title("Blink Detection with Face Mesh")

# Load custom sound
audio_file = "alert.mp3"

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define eyes landmark indices for EAR calculation
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Helper functions
def euclidean(p1, p2):
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

def eye_aspect_ratio(landmarks, eye_indices):
    p = [landmarks[i] for i in eye_indices]
    return (euclidean(p[1], p[5]) + euclidean(p[2], p[4])) / (2 * euclidean(p[0], p[3]))

# Webcam input via Streamlit
webrtc = st.camera_input("Look into the camera")

if webrtc is not None:
    # Convert to BGR (OpenCV format)
    frame = np.array(webrtc)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Process frame
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Draw landmarks
        for lm in landmarks:
            x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Compute EAR
        ear = (eye_aspect_ratio(landmarks, LEFT_EYE) + eye_aspect_ratio(landmarks, RIGHT_EYE)) / 2

        # Blink detection threshold
        if ear < 0.25:
            st.audio(audio_file)  # Play alert while eyes are closed

    # Display frame
    st.image(frame, channels="BGR")
