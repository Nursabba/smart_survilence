import streamlit as st
from ultralytics import YOLO
import cv2, csv, os, time
from datetime import datetime
import tempfile

st.set_page_config(page_title="AI Smart Surveillance")

model = YOLO("yolov8n.pt")

os.makedirs("captures", exist_ok=True)
os.makedirs("recordings", exist_ok=True)

log_file = "surveillance_log.csv"
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(
            ["Time","Persons","Cars","Bikes","Alert"]
        )

video = st.file_uploader(
    "Upload CCTV Video",
    type=["mp4","avi","mov"]
)
if video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video.read())

    cap = cv2.VideoCapture(tfile.name)

    video_name = "recordings/" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_name, fourcc, 20, (640,480))

    RX1,RY1,RX2,RY2 = 180,120,460,420
    frame_box = st.image([])
    last_capture = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame,(640,480))
        results = model(frame, verbose=False)[0]

        persons=cars=bikes=0
        alert="NORMAL"

        cv2.rectangle(frame,(RX1,RY1),(RX2,RY2),(0,0,255),2)

        for box in results.boxes:
            cls=int(box.cls[0])
            conf=float(box.conf[0])

            if cls not in [0,2,3]:
                continue

            x1,y1,x2,y2=map(int,box.xyxy[0])
            cx,cy=(x1+x2)//2,(y1+y2)//2

            if cls==0:
                persons+=1
                label="Person"
                color=(0,255,0)
            elif cls==2:
                cars+=1
                label="Car"
                color=(255,0,0)
            else:
                bikes+=1
                label="Bike"
                color=(0,255,255)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"{label} {conf:.2f}",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

            if cls==0 and RX1<cx<RX2 and RY1<cy<RY2:
                alert="INTRUSION ALERT"

        if alert=="INTRUSION ALERT" and time.time()-last_capture>5:
            img="captures/"+datetime.now().strftime("%Y%m%d_%H%M%S")+".jpg"
            cv2.imwrite(img,frame)
            last_capture=time.time()

        now=datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        with open(log_file,"a",newline="") as f:
            csv.writer(f).writerow([now,persons,cars,bikes,alert])

        out.write(frame)

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame_box.image(rgb)

    out.release()
    cap.release()
    st.success("Surveillance Completed")
