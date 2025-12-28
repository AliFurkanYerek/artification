import cv2
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def save_results(script_name: str, video_source: str, giren: int, cikan: int) -> None:
    """Sayim sonucunu JSON olarak yaz."""
    payload = {
        "script": script_name,
        "video_source": video_source,
        "giren": giren,
        "cikan": cikan,
        "fark": giren - cikan,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    (DATA_DIR / f"{script_name}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

#modeli tanımlama
model = YOLO("yolov8n.pt")

#video capture
VIDEO_SOURCE = "1.mp4"
cap = cv2.VideoCapture(VIDEO_SOURCE)

success, frame = cap.read()
if not success:
    exit("Video acilamadi")

#yeniden boyut
frame = cv2.resize(frame,(0,0), fx= 0.6 , fy=0.6)
frame_height, frame_width = frame.shape[:2]     



line_x=int (frame_width *0.5)
offset=10


#sayaclar

giren=0
cikan=0
counted_ids= set()
person_last_x ={}
save_results("main_street", VIDEO_SOURCE, giren, cikan)

#yolo insan ile sayma 
while True:
    success, frame = cap.read()
    if not success:
        break

    counts_changed = False
    frame = cv2.resize(frame,(0,0), fx= 0.6 , fy=0.6)
    results = model.track(frame, persist=True, stream=False, conf = 0.25, iou = 0.3, tracker = "bytetrack.yaml")

    if results[0].boxes.id is not None:
        ids=results[0].boxes.id.int().tolist()
        classes=results[0].boxes.cls.int().tolist()
        xyxy = results[0].boxes.xyxy
        for i, box in enumerate(xyxy):
            cls_id=classes[i]
            track_id=ids[i]
            class_name=model.names[cls_id]
            if class_name != "person":
                continue
            x1,y1,x2,y2= map(int, box)
            cx=int((x1+x2)/2)
            cy= int((y1+y2)/2)
            previous_x = person_last_x.get(track_id,None)
            person_last_x[track_id] =cx

            if previous_x is not None:
                if previous_x < line_x <=cx:
                    if track_id not in counted_ids:
                        cikan+=1
                        counted_ids.add(track_id)
                        counts_changed = True

                
                elif previous_x > line_x >= cx:
                    if track_id not in counted_ids:
                        giren+=1
                        counted_ids.add(track_id)
                        counts_changed = True
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1,y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.circle(frame, (cx, cy), 4, (255,0,0), -1)

    cv2.line(frame,(line_x,0),(line_x,frame_height),(0,0,255),2)

    cv2.putText(frame, f"Giren (saga): {giren}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(frame, f"Cikan (sola): {cikan}",(10,60),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    if counts_changed:
        save_results("main_street", VIDEO_SOURCE, giren, cikan)
    cv2.imshow("avm yon takibi", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
save_results("main_street", VIDEO_SOURCE, giren, cikan)
cv2.destroyAllWindows()