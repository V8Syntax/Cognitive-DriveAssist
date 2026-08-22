import cv2
import mediapipe as mp
import urllib.request
import os
import time
import math

# --- EAR MATH FUNCTION ---
def calculate_ear(eye_landmarks):
    # Calculate the vertical distances
    v1 = math.dist([eye_landmarks[1].x, eye_landmarks[1].y], [eye_landmarks[5].x, eye_landmarks[5].y])
    v2 = math.dist([eye_landmarks[2].x, eye_landmarks[2].y], [eye_landmarks[4].x, eye_landmarks[4].y])
    
    # Calculate the horizontal distance
    h = math.dist([eye_landmarks[0].x, eye_landmarks[0].y], [eye_landmarks[3].x, eye_landmarks[3].y])
    
    # Return the EAR ratioc
    if h == 0:
        return 0
    return (v1 + v2) / (2.0 * h)

# --- SETUP AI MODEL ---
model_path = 'face_landmarker.task'
options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_faces=1
)

# --- SETTINGS ---
EAR_THRESHOLD = 0.20    # If EAR drops below this, eyes are considered closed
FRAMES_TO_WAIT = 20     # Number of consecutive frames eyes must be closed to trigger alarm

# MediaPipe indices for the 6 points of the Right and Left eyes
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

print("Starting Phase 2 Brain...")
cap = cv2.VideoCapture(0)
closed_frames_counter = 0

with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            
            # Extract just the eye landmarks
            right_eye = [landmarks[i] for i in RIGHT_EYE_INDICES]
            left_eye = [landmarks[i] for i in LEFT_EYE_INDICES]
            
            # Calculate EAR for both eyes and average them
            right_ear = calculate_ear(right_eye)
            left_ear = calculate_ear(left_eye)
            average_ear = (right_ear + left_ear) / 2.0

            # --- THE LOGIC ---
            if average_ear < EAR_THRESHOLD:
                closed_frames_counter += 1
            else:
                closed_frames_counter = 0
                
            # --- THE ALERT ---
            if closed_frames_counter >= FRAMES_TO_WAIT:
                cv2.putText(frame, "DROWSINESS DETECTED!", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            # (Optional) Print EAR on screen so you can see the math working
            cv2.putText(frame, f"EAR: {average_ear:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Cognitive DriveAssist - Phase 2', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()