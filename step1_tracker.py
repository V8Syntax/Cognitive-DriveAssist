import cv2
import mediapipe as mp
import urllib.request
import os
import time

# 1. Download the AI Model automatically (only happens the very first time)
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading the modern AI model for the first time. Please wait a few seconds...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# 2. Setup the new MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

print("Starting webcam...")
cap = cv2.VideoCapture(0)

# 3. Run the live video loop
with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # MediaPipe needs the image in RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Create a millisecond timestamp for the video frame
        timestamp_ms = int(time.time() * 1000)

        # Detect the face landmarks
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Draw the 468 points directly onto your face
        if result.face_landmarks:
            for face_landmarks in result.face_landmarks:
                for landmark in face_landmarks:
                    h, w, _ = frame.shape
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    # Draw a tiny green dot at each coordinate
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Show the desktop window
        cv2.imshow('Cognitive DriveAssist - Phase 1', frame)

        # Press 'q' to quit the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()