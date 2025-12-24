import cv2
import mediapipe as mp
import math
import pygame


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# EAR threshold
EYE_THRESHOLD = 0.25


pygame.mixer.init()
pygame.mixer.music.load("audio.mp3")  


sound_playing = False


def euclidean(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def eye_aspect_ratio(landmarks, eye):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye]
    return (euclidean(p2, p6) + euclidean(p3, p5)) / (2 * euclidean(p1, p4))


while True:
    ret, frame = cap.read()
    if not ret:
        break


    frame = cv2.flip(frame, 1)


    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]


        mp_drawing.draw_landmarks(
            frame,
            landmarks,
            mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1)
        )

    
        for idx in LEFT_EYE + RIGHT_EYE:
            x = int(landmarks.landmark[idx].x * frame.shape[1])
            y = int(landmarks.landmark[idx].y * frame.shape[0])
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        left_ear = eye_aspect_ratio(landmarks.landmark, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks.landmark, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2


        if ear < EYE_THRESHOLD:
            if not sound_playing:
                pygame.mixer.music.play(-1) 
                sound_playing = True
        else:
            if sound_playing:
                pygame.mixer.music.stop()
                sound_playing = False

    
    cv2.imshow("Face Mesh + Blink Detection", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
