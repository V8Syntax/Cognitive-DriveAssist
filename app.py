import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import sqlite3
import datetime
import pandas as pd
import math
import time
import os
import winsound 
import wave
import struct

# --- 1. GENERATE OUR OWN LOUD ALARM ---
# This creates a piercing 2500Hz siren file on your computer so we don't have to download one!
alarm_path = "loud_siren.wav"

def create_siren():
    if not os.path.exists(alarm_path):
        sample_rate = 44100
        duration = 1.0 # 1 second long
        frequency = 2500.0 # Very high-pitched piercing tone
        
        with wave.open(alarm_path, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            for i in range(int(sample_rate * duration)):
                # Generate maximum volume math sine wave
                value = int(32767.0 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
                data = struct.pack('<h', value)
                wav_file.writeframesraw(data)

create_siren()

# --- 2. DATABASE INITIALIZATION ---
DB_NAME = "driveassist.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fatigue_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            ear REAL,
            status TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_event(ear, status="Drowsiness Detected"):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO fatigue_logs (timestamp, ear, status)
        VALUES (?, ?, ?)
    """, (current_time, round(ear, 2), status))
    conn.commit()
    conn.close()

def get_logs():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT timestamp AS 'Timestamp', ear AS 'EAR Score', status AS 'Event Status' FROM fatigue_logs ORDER BY id DESC LIMIT 10", conn)
    conn.close()
    return df

init_db()

# --- 3. EAR CALCULATION HELPER ---
def calculate_ear(eye_landmarks):
    v1 = math.dist([eye_landmarks[1].x, eye_landmarks[1].y], [eye_landmarks[5].x, eye_landmarks[5].y])
    v2 = math.dist([eye_landmarks[2].x, eye_landmarks[2].y], [eye_landmarks[4].x, eye_landmarks[4].y])
    h = math.dist([eye_landmarks[0].x, eye_landmarks[0].y], [eye_landmarks[3].x, eye_landmarks[3].y])
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)

# --- 4. PAGE CONFIGURATION ---
st.set_page_config(page_title="Cognitive DriveAssist Dashboard", layout="wide")

st.title("🚗 Cognitive DriveAssist Monitor")
st.markdown("Real-time facial geometry tracking and driver alertness monitoring pipeline.")

# Sidebar Configuration
st.sidebar.header("⚙️ System Configuration")
ear_threshold = st.sidebar.slider("EAR Threshold", min_value=0.10, max_value=0.35, value=0.20, step=0.01)
frames_to_wait = st.sidebar.slider("Consecutive Closed Frames Trigger", min_value=5, max_value=60, value=10, step=1)

run_monitoring = st.sidebar.toggle("Start Driver Monitoring", value=False)

if st.sidebar.button("Clear Log History"):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM fatigue_logs")
    conn.commit()
    conn.close()
    st.sidebar.success("Database cleared.")

# Main Layout: 2 Columns
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📹 Live Driver Feed")
    video_placeholder = st.empty()

with col2:
    st.subheader("📊 Live Telemetry & Alerts")
    status_placeholder = st.empty()
    metric_col1, metric_col2 = st.columns(2)
    ear_metric_placeholder = metric_col1.empty()
    frame_metric_placeholder = metric_col2.empty()
    
    st.markdown("---")
    st.subheader("📋 Recent Incident Database Logs")
    table_placeholder = st.empty()

# --- 5. CORE EXECUTION PIPELINE ---
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
model_path = 'face_landmarker.task'

if run_monitoring:
    cap = cv2.VideoCapture(0)
    closed_frames_counter = 0
    alert_logged = False

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1
    )

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened() and run_monitoring:
            success, frame = cap.read()
            if not success:
                st.error("Failed to access webcam feed.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            current_ear = 0.0

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                right_eye = [landmarks[i] for i in RIGHT_EYE_INDICES]
                left_eye = [landmarks[i] for i in LEFT_EYE_INDICES]

                r_ear = calculate_ear(right_eye)
                l_ear = calculate_ear(left_eye)
                current_ear = (r_ear + l_ear) / 2.0

                # Drowsiness Decision Logic
                if current_ear < ear_threshold:
                    closed_frames_counter += 1
                else:
                    closed_frames_counter = 0
                    alert_logged = False

                # Handle Alert State
                if closed_frames_counter >= frames_to_wait:
                    # Visual Alert Overlay
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 8)
                    cv2.putText(frame, "DROWSINESS DETECTED!", (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    
                    status_placeholder.error("⚠️ CRITICAL ALERT: Driver Fatigue Detected!")
                    
                    # Log event to database and play sound EXACTLY ONCE per incident
                    if not alert_logged:
                        winsound.PlaySound(alarm_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                        log_event(current_ear, "Drowsiness Alert")
                        alert_logged = True
                else:
                    status_placeholder.success("✅ Driver State: Attentive / Active")
            else:
                status_placeholder.warning("⚠️ No Driver Face Detected")
                closed_frames_counter = 0

            # Update Live Visuals
            frame_rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb_display, channels="RGB", use_container_width=True)

            # Update Metrics & Table
            ear_metric_placeholder.metric("Current EAR", f"{current_ear:.2f}")
            frame_metric_placeholder.metric("Closed Frames", f"{closed_frames_counter}/{frames_to_wait}")
            table_placeholder.dataframe(get_logs(), use_container_width=True)

    cap.release()
else:
    status_placeholder.info("Monitoring is paused. Toggle 'Start Driver Monitoring' in the sidebar to begin.")
    table_placeholder.dataframe(get_logs(), use_container_width=True)