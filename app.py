import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile

st.title("AI Smart Surveillance System")

model = YOLO("yolov8n.pt")
video = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

if video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video.read())

    cap = cv2.VideoCapture(tfile.name)
    frame_view = st.image([])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            if cls == 0:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_view.image(frame)

    cap.release()

    st.success("Surveillance completed")
