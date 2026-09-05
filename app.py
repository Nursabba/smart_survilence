import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import cv2

st.title("AI Smart Surveillance System")
model = YOLO("yolov8n.pt")

class Detector(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, verbose=False)[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            if cls in [0,2,3]:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

        return frame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="surveillance",
    video_processor_factory=Detector,
    media_stream_constraints={"video": True, "audio": False},
)


