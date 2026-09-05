import streamlit as st
import cv2
import mediapipe as mp
import tempfile

st.title("AI Smart Surveillance System")

video = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

if video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video.read())

    cap = cv2.VideoCapture(tfile.name)
    frame_window = st.image([])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        frame_window.image(rgb)

    cap.release()
    st.success("Surveillance completed")