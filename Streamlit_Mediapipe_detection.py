import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import torch
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import tempfile

st.set_page_config(page_title="Live Detection Dashboard", page_icon="👁️")

# Initialize MediaPipe Holistic model for Tab 5 (Facial and Hand Landmarks)
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize MediaPipe Hands for Tab 6 (Gesture Recognition)
mp_hands = mp.solutions.hands
hands_model = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize drawing utils for MediaPipe
mp_drawing = mp.solutions.drawing_utils

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load YOLOv5 model for Object Detection (Tab 4)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define RTC configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# VideoProcessor for Facial and Hand Landmarks (Tab 5)
class HolisticVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.previousTime = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (800, 600))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = holistic_model.process(img_rgb)
        img_rgb.flags.writeable = True
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Draw facial landmarks
        mp_drawing.draw_landmarks(
            img,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
        )

        # Draw hand landmarks
        mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Display FPS
        currentTime = time.time()
        fps = 1 / (currentTime - self.previousTime) if self.previousTime != 0 else 0
        self.previousTime = currentTime
        cv2.putText(img, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# VideoProcessor for Face and Eye Detection (Tab 3)
class FaceEyeVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.previousTime = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (800, 600))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        currentTime = time.time()
        fps = 1 / (currentTime - self.previousTime) if self.previousTime != 0 else 0
        self.previousTime = currentTime
        cv2.putText(img, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# VideoProcessor for Gesture Recognition (Tab 6)
class GestureVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.previousTime = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (800, 600))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = hands_model.process(img_rgb)
        img_rgb.flags.writeable = True
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        gesture = "None"
        debug_info = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks with indices for debugging
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.putText(img, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmarks
                landmarks = hand_landmarks.landmark
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                middle_tip = landmarks[12]
                ring_tip = landmarks[16]
                pinky_tip = landmarks[20]
                index_pip = landmarks[6]
                middle_pip = landmarks[10]
                ring_pip = landmarks[14]
                pinky_pip = landmarks[18]
                thumb_ip = landmarks[2]

                # Calculate distances
                def distance(p1, p2):
                    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

                index_dist = distance(index_tip, index_pip)
                middle_dist = distance(middle_tip, middle_pip)
                ring_dist = distance(ring_tip, ring_pip)
                pinky_dist = distance(pinky_tip, pinky_pip)
                thumb_dist = distance(thumb_tip, thumb_ip)
                index_middle_dist = distance(index_tip, middle_tip)

                # Thumbs Up: Thumb tip above fingers, others folded
                thumb_up = (
                    thumb_tip.y < index_tip.y - 0.05 and
                    thumb_tip.y < middle_tip.y - 0.05 and
                    thumb_tip.y < ring_tip.y - 0.05 and
                    thumb_tip.y < pinky_tip.y - 0.05 and
                    index_dist < 0.06 and
                    middle_dist < 0.06 and
                    ring_dist < 0.06 and
                    pinky_dist < 0.06 and
                    abs(thumb_tip.x - thumb_ip.x) < 0.05  # Thumb vertical
                )
                debug_info.append(f"Thumb Up: {thumb_up}")

                # Victory Sign: Index and middle extended, others folded, fingers separated
                victory_sign = (
                    index_dist > 0.12 and
                    middle_dist > 0.12 and
                    ring_dist < 0.06 and
                    pinky_dist < 0.06 and
                    thumb_dist < 0.06 and
                    index_middle_dist > 0.08 and  # Fingers separated
                    abs(index_tip.y - middle_tip.y) < 0.06  # Tips aligned
                )
                debug_info.append(f"Victory Sign: {victory_sign}")

                # Closed Fist: All fingers folded
                closed_fist = (
                    index_dist < 0.06 and
                    middle_dist < 0.06 and
                    ring_dist < 0.06 and
                    pinky_dist < 0.06 and
                    thumb_dist < 0.06
                )
                debug_info.append(f"Closed Fist: {closed_fist}")

                # Open Hand: All fingers extended
                open_hand = (
                    index_dist > 0.12 and
                    middle_dist > 0.12 and
                    ring_dist > 0.12 and
                    pinky_dist > 0.12 and
                    thumb_dist > 0.12
                )
                debug_info.append(f"Open Hand: {open_hand}")

                # Assign gesture
                if thumb_up:
                    gesture = "Thumbs Up"
                elif victory_sign:
                    gesture = "Victory Sign"
                elif closed_fist:
                    gesture = "Closed Fist"
                elif open_hand:
                    gesture = "Open Hand"

        # Display gesture, FPS, and debug info
        currentTime = time.time()
        fps = 1 / (currentTime - self.previousTime) if self.previousTime != 0 else 0
        self.previousTime = currentTime
        cv2.putText(img, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"Gesture: {gesture}", (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        for i, info in enumerate(debug_info):
            cv2.putText(img, info, (10, 130 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Custom CSS for attractive frontend with visible black text
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: #000000;
        font-family: 'Arial', sans-serif;
    }
    .main {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        max-width: 900px;
        margin: auto;
        
    }
    .stApp {
        background: transparent;
    }
    .app-title {
        font-size: 3em;
        color: #000000;
        text-align: center;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    h1 {
        font-size: 2.5em;
        color: #000000;
        text-align: center;
        margin-bottom: 10px;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    .description {
        font-size: 1.1em;
        color: #000000;
        text-align: center;
        margin-bottom: 5px;
    }
    .note {
        font-size: 0.9em;
        color: #000000;
        text-align: center;
        margin-top: 10px;
    }
    .stButton > button {
        background-color: #e53e3e;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 1em;
        cursor: pointer;
        transition: background-color 0.3s;
        display: block;
        margin: 10px auto;
    }
    .stButton > button:hover {
        background-color: #c53030;
    }
    .stSidebar {
        background: #2a5298;
        border-radius: 10px;
        padding: 10px;
    }
    .stSidebar .css-1d391kg {
        color: #000000;
    }
    .stSidebar .css-1d391kg:hover {
        background-color: #1e3c72;
        color: #000000;
    }
    .webrtc-container {
        border: 2px solid #2a5298;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# App-wide title
st.markdown("<div class='app-title'>Live Detection Dashboard</div>", unsafe_allow_html=True)

# Sidebar for tab navigation
st.sidebar.title("Detection Modes")
tab = st.sidebar.radio(
    "Select Detection Mode",
    [
        "Face and Eye Detection (Photo)",
        "Face and Eye Detection (Video)",
        "Face and Eye Detection (Live Camera)",
        "Object Detection",
        "Facial and Hand Landmarks Detection",
        "Gesture Recognition"
    ]
)

# Main container
with st.container():
    st.markdown("<div class='main'>", unsafe_allow_html=True)

    # Tab 1: Face and Eye Detection using Photo
    if tab == "Face and Eye Detection (Photo)":
        st.markdown("<h1>Face and Eye Detection (Photo)</h1>", unsafe_allow_html=True)
        st.markdown(
            "<div class='description'>Upload an image to detect faces and eyes using Haar cascades.</div>",
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (800, 600))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            st.image(img, channels="BGR", caption="Processed Image")
        st.markdown(
            "<div class='note'>Note: Upload a clear image with visible faces for best results.</div>",
            unsafe_allow_html=True
        )

    # Tab 2: Face and Eye Detection using Video
    elif tab == "Face and Eye Detection (Video)":
        st.markdown("<h1>Face and Eye Detection (Video)</h1>", unsafe_allow_html=True)
        st.markdown(
            "<div class='description'>Upload a video to detect faces and eyes using Haar cascades.</div>",
            unsafe_allow_html=True
        )
        uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (800, 600))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = frame[y:y+h, x:x+w]
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

                stframe.image(frame, channels="BGR")
            cap.release()
        st.markdown(
            "<div class='note'>Note: Upload a short video for faster processing.</div>",
            unsafe_allow_html=True
        )

    # Tab 3: Face and Eye Detection using Live Camera
    elif tab == "Face and Eye Detection (Live Camera)":
        st.markdown("<h1>Face and Eye Detection (Live Camera)</h1>", unsafe_allow_html=True)
        st.markdown(
            "<div class='description'>Use your webcam for real-time face and eye detection.</div>",
            unsafe_allow_html=True
        )
        st.markdown("<div class='webrtc-container'>", unsafe_allow_html=True)
        webrtc_streamer(
            key="face-eye-detection",
            video_processor_factory=FaceEyeVideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False}
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='note'>Note: Allow webcam access to start detection.</div>",
            unsafe_allow_html=True
        )

    # Tab 4: Object Detection
    elif tab == "Object Detection":
        st.markdown("<h1>Object Detection</h1>", unsafe_allow_html=True)
        st.markdown(
            "<div class='description'>Upload an image to detect objects using YOLOv5.</div>",
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (800, 600))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Run YOLOv5 inference
            results = yolo_model(img_rgb)
            results.render()  # Draw bounding boxes and labels
            img_with_detections = results.ims[0]  # Get the rendered image
            img_with_detections = cv2.cvtColor(img_with_detections, cv2.COLOR_RGB2BGR)

            st.image(img_with_detections, channels="BGR", caption="Processed Image with Object Detections")
        st.markdown(
            "<div class='note'>Note: Upload a clear image with visible objects for best results.</div>",
            unsafe_allow_html=True
        )

    # Tab 5: Facial and Hand Landmarks Detection
    elif tab == "Facial and Hand Landmarks Detection":
        st.markdown("<h1>Facial & Hand Landmarks Detection</h1>", unsafe_allow_html=True)
        st.markdown(
            "<div class='description'>Detect facial and hand landmarks in real-time using MediaPipe.</div>",
            unsafe_allow_html=True
        )
        st.markdown("<div class='webrtc-container'>", unsafe_allow_html=True)
        webrtc_streamer(
            key="face-hand-detection",
            video_processor_factory=HolisticVideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False}
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='note'>Note: Allow webcam access to start detection.</div>",
            unsafe_allow_html=True
        )

    # Tab 6: Gesture Recognition
    else:
        st.markdown("<h1>Gesture Recognition</h1>", unsafe_allow_html=True)
        st.markdown(
            "<div class='description'>Use your webcam to recognize hand gestures in real-time using MediaPipe.</div>",
            unsafe_allow_html=True
        )
        st.markdown("<div class='webrtc-container'>", unsafe_allow_html=True)
        webrtc_streamer(
            key="gesture-detection",
            video_processor_factory=GestureVideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False}
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='note'>Note: Allow webcam access and show clear hand gestures (e.g., thumbs up, victory sign, open hand, closed fist). Ensure good lighting and hand visibility.</div>",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)