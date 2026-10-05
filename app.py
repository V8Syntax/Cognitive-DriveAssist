import datetime
import html
import math
import os
import sqlite3
import struct
import time
import wave

import cv2
import mediapipe as mp
import pandas as pd
import streamlit as st
import winsound


# =========================================================
# DriveAssist page and soft UI
# =========================================================
st.set_page_config(
    page_title="DriveAssist | Fleet Safety Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=JetBrains+Mono:wght@600&display=swap');
    .stApp { background:#F4F5F8 !important; color:#172033; font-family:'Plus Jakarta Sans',sans-serif; }
    #MainMenu, footer, header { visibility:hidden; }
    .block-container { padding-top:1.5rem !important; padding-bottom:2.5rem; max-width:1400px; }
    [data-testid="stVerticalBlockBorderWrapper"] { background:#fff !important; border:1px solid rgba(226,232,240,.85) !important; border-radius:22px; padding:20px 24px; box-shadow:0 10px 25px -4px rgba(15,23,42,.05); }
    .topbar { display:flex; justify-content:space-between; align-items:center; margin-bottom:1.3rem; }
    .brand { color:#172033; font:800 1.45rem 'Plus Jakarta Sans',sans-serif; letter-spacing:-.04em; }
    .subbrand { color:#7b8495; font-size:.82rem; margin-top:.25rem; }
    .system-pill { display:inline-flex; align-items:center; gap:.5rem; padding:.45rem .8rem; border:1px solid #c9efe4; border-radius:999px; background:#e8f8f3; color:#24856d; font-size:.78rem; }
    .system-dot { width:7px;height:7px;border-radius:50%;background:#39b894; }
    .section-title { color:#202b3d; font:700 1rem 'Plus Jakarta Sans',sans-serif; margin:.15rem 0 .85rem; }
    .section-note { color:#8992a2; font-size:.76rem; margin-top:-.45rem; margin-bottom:.8rem; }
    .avatar { width:60px;height:60px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#e2f5ef;color:#278a72;font:800 1.1rem 'Plus Jakarta Sans',sans-serif; }
    .driver-name { color:#172033;font:800 1.2rem 'Plus Jakarta Sans',sans-serif;margin-top:.75rem; }
    .driver-role { color:#8791a1;font-size:.78rem;margin:.2rem 0 1rem; }
    .vehicle { color:#253149;font-weight:700;font-size:.9rem;padding:.8rem 0;border-top:1px solid #edf0f4; }
    .detail-row { display:flex;justify-content:space-between;color:#8590a0;font-size:.78rem;padding:.42rem 0; }
    .detail-row strong { color:#344158;font-weight:700; }
    .mini-card { background:#f7f9fc;border:1px solid #edf0f4;border-radius:15px;padding:12px 14px;min-height:78px;margin-bottom:.8rem; }
    .mini-label { color:#8992a2;font-size:.66rem;text-transform:uppercase;letter-spacing:.06em;font-weight:700; }
    .mini-value { color:#243149;font:700 1.15rem 'JetBrains Mono',monospace;margin-top:.35rem; }
    .incident { display:flex;gap:10px;align-items:flex-start;padding:12px 0;border-bottom:1px solid #edf0f4; }
    .incident:last-child { border-bottom:0; }
    .incident-icon { width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#fff1e7;font-size:15px;flex:0 0 auto; }
    .incident-title { color:#303b4e;font-size:.76rem;font-weight:700; }
    .incident-time { color:#9aa2af;font-size:.66rem;margin-top:4px; }
    .ear-pill { margin-left:auto;background:#fff5d8;color:#9a761b;border-radius:999px;padding:4px 8px;font:600 .63rem 'JetBrains Mono',monospace;white-space:nowrap; }
    [data-testid="stMetric"] { background:#f7f9fc;border:1px solid #edf0f4;border-radius:13px;padding:10px 12px; }
    [data-testid="stMetricLabel"] { color:#7d8798; }
    .stAlert { border-radius:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


DB_NAME = "driveassist.db"
MODEL_PATH = "face_landmarker.task"
ALARM_FILE = "loud_siren.wav"
EAR_THRESHOLD = 0.23
CONSECUTIVE_FRAMES = 20
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# =========================================================
# Audio and database helpers
# =========================================================
def generate_siren_wav(filename=ALARM_FILE):
    if os.path.exists(filename):
        return
    sample_rate, duration, frequency = 44100, 0.5, 2500.0
    with wave.open(filename, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for index in range(int(sample_rate * duration)):
            value = int(32767 * 0.7 * math.sin(2 * math.pi * frequency * index / sample_rate))
            wav.writeframesraw(struct.pack("<h", value))


def trigger_audio_alarm():
    try:
        winsound.PlaySound(ALARM_FILE, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except (RuntimeError, OSError) as exc:
        st.warning(f"Audio alert could not be played: {exc}")


def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS fatigue_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                ear REAL,
                status TEXT
            )"""
        )


def log_event(ear, status="Drowsiness Alert"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute(
            "INSERT INTO fatigue_logs (timestamp, ear, status) VALUES (?, ?, ?)",
            (timestamp, round(float(ear), 3), status),
        )


def fetch_events(limit=5):
    with sqlite3.connect(DB_NAME) as conn:
        return pd.read_sql_query(
            """SELECT timestamp AS timestamp, ear AS ear_value, status AS event_type
               FROM fatigue_logs ORDER BY id DESC LIMIT ?""",
            conn,
            params=(int(limit),),
        )


def clear_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("DELETE FROM fatigue_logs")


def build_incidents_html():
    events = fetch_events(5)
    if events.empty:
        return '<div class="section-note">No incidents recorded. Driver is ready for monitoring.</div>'
    items = []
    for _, row in events.iterrows():
        status = html.escape(str(row["event_type"]))
        timestamp = html.escape(str(row["timestamp"]))
        items.append(
            '<div class="incident"><div class="incident-icon">⚠️</div>'
            f'<div><div class="incident-title">{status}</div>'
            f'<div class="incident-time">{timestamp}</div></div>'
            f'<span class="ear-pill">EAR {float(row["ear_value"]):.2f}</span></div>'
        )
    return "".join(items)


def calculate_ear(landmarks, indices):
    points = [landmarks[index] for index in indices]
    distance = lambda a, b: math.hypot(a.x - b.x, a.y - b.y)
    horizontal = distance(points[0], points[3])
    if horizontal == 0:
        return 0.0
    return (distance(points[1], points[5]) + distance(points[2], points[4])) / (2 * horizontal)


def open_camera_and_landmarker():
    if not os.path.exists(MODEL_PATH):
        return False, f"MediaPipe model not found: {MODEL_PATH}"

    capture = None
    for camera_index in (0, 1):
        for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
            candidate = cv2.VideoCapture(camera_index, backend)
            if candidate.isOpened():
                capture = candidate
                break
            candidate.release()
        if capture is not None:
            break

    if capture is None:
        return False, "Could not open the camera. Close Teams or other camera apps, then toggle Live Stream off and on."

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
    )
    try:
        landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    except Exception as exc:
        capture.release()
        return False, f"Could not start the face tracker: {exc}"

    st.session_state.camera_capture = capture
    st.session_state.face_landmarker = landmarker
    st.session_state.closed_frame_count = 0
    st.session_state.alert_logged = False
    st.session_state.last_video_timestamp = 0
    st.session_state.last_frame_time = None
    return True, None


def close_camera():
    landmarker = st.session_state.pop("face_landmarker", None)
    capture = st.session_state.pop("camera_capture", None)
    if landmarker is not None:
        landmarker.close()
    if capture is not None:
        capture.release()


init_db()
generate_siren_wav()


@st.fragment(run_every=0.1)
def live_camera_fragment():
    capture = st.session_state.get("camera_capture")
    landmarker = st.session_state.get("face_landmarker")
    if capture is None or landmarker is None or not capture.isOpened():
        st.error("Camera is not connected. Turn Live Stream off and on to retry.")
        return

    success, frame = capture.read()
    if not success:
        st.error("The camera opened but did not return a frame. Check camera permissions and close other camera apps.")
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = max(int(time.time() * 1000), st.session_state.last_video_timestamp + 1)
    st.session_state.last_video_timestamp = timestamp_ms
    result = landmarker.detect_for_video(media_image, timestamp_ms)

    ear = None
    closed_frames = st.session_state.closed_frame_count
    alert_logged = st.session_state.alert_logged
    attention = "No face detected"
    if result.face_landmarks:
        landmarks = result.face_landmarks[0]
        ear = (calculate_ear(landmarks, LEFT_EYE) + calculate_ear(landmarks, RIGHT_EYE)) / 2
        for index in LEFT_EYE + RIGHT_EYE:
            x = int(landmarks[index].x * frame.shape[1])
            y = int(landmarks[index].y * frame.shape[0])
            cv2.circle(rgb, (x, y), 2, (16, 185, 129), -1)

        if ear < EAR_THRESHOLD:
            closed_frames += 1
            attention = "Eyes closing"
        else:
            closed_frames = 0
            alert_logged = False
            attention = "Attentive / Active"

        if closed_frames >= CONSECUTIVE_FRAMES:
            attention = "Drowsiness detected"
            cv2.rectangle(rgb, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (239, 68, 68), 6)
            cv2.putText(rgb, "ALERT: DROWSINESS DETECTED", (25, 85),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (239, 68, 68), 2)
            if not alert_logged:
                trigger_audio_alarm()
                log_event(ear, "Drowsiness Alert")
                alert_logged = True
                st.session_state.alert_logged = True
                st.session_state.closed_frame_count = closed_frames
                st.rerun(scope="app")
    else:
        closed_frames = 0
        alert_logged = False

    st.session_state.closed_frame_count = closed_frames
    st.session_state.alert_logged = alert_logged
    now = time.monotonic()
    previous = st.session_state.last_frame_time
    fps = 1 / (now - previous) if previous is not None and now > previous else 0
    st.session_state.last_frame_time = now

    cv2.rectangle(rgb, (20, 20), (190, 60), (255, 255, 255), -1)
    cv2.putText(rgb, f"EAR: {ear:.3f}" if ear is not None else "EAR: --",
                (35, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (15, 23, 42), 2)

    info_cols = st.columns(3)
    info_cols[0].metric("Driver attention", attention)
    info_cols[1].metric("Current EAR", f"{ear:.3f}" if ear is not None else "—")
    info_cols[2].metric("Closed frames", f"{closed_frames}/{CONSECUTIVE_FRAMES}")
    st.caption(f"Camera inference: {fps:.1f} FPS · EAR fatigue limit: {EAR_THRESHOLD:.2f}")
    st.image(rgb, channels="RGB", use_container_width=True)


# =========================================================
# Header and dashboard cards
# =========================================================
st.markdown(
    """
    <div class="topbar">
      <div><div class="brand">DriveAssist</div>
      <div class="subbrand">Driver safety active &nbsp;·&nbsp; Unit #04-Alpha</div></div>
      <div class="system-pill"><i class="system-dot"></i> System operational</div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_left, col_mid, col_right = st.columns([3, 4, 3], gap="large")

with col_left:
    with st.container(border=True):
        st.markdown(
            '<div class="section-title">Driver &amp; vehicle</div>'
            '<div class="avatar">K</div>'
            '<div class="driver-name">Ketal</div>'
            '<div class="driver-role">Fleet driver · Unit #04-Alpha</div>'
            '<div class="vehicle">Porsche 911 Carrera S</div>'
            '<div class="detail-row"><span>Battery level</span><strong>86%</strong></div>'
            '<div class="detail-row"><span>Hardware status</span><strong>Online</strong></div>'
            '<div class="detail-row"><span>Persistence</span><strong>SQLite</strong></div>',
            unsafe_allow_html=True,
        )

with col_mid:
    with st.container(border=True):
        st.markdown('<div class="section-title">Live camera &amp; attention</div>', unsafe_allow_html=True)
        speed_col, ear_col = st.columns(2)
        with speed_col:
            st.markdown('<div class="mini-card"><div class="mini-label">Inference speed</div><div class="mini-value">Live</div></div>', unsafe_allow_html=True)
        with ear_col:
            st.markdown('<div class="mini-card"><div class="mini-label">EAR limit</div><div class="mini-value">0.23</div></div>', unsafe_allow_html=True)

        run_cam = st.toggle("Activate Live Stream", value=False, key="run_cam")
        if run_cam:
            camera_ready = st.session_state.get("camera_capture") is not None
            if not camera_ready:
                camera_ready, camera_error = open_camera_and_landmarker()
                if not camera_ready:
                    st.error(camera_error)
            if camera_ready:
                live_camera_fragment()
        else:
            close_camera()
            st.info("Monitoring is paused. Activate Live Stream to begin.")

with col_right:
    with st.container(border=True):
        title_col, clear_col = st.columns([2, 1])
        with title_col:
            st.markdown('<div class="section-title">Recent incidents</div>', unsafe_allow_html=True)
        with clear_col:
            if st.button("Clear Log", use_container_width=True):
                clear_db()
                st.rerun()
        st.markdown('<div class="section-note">Driver safety event history</div>', unsafe_allow_html=True)
        incidents_container = st.empty()
        incidents_container.markdown(build_incidents_html(), unsafe_allow_html=True)
