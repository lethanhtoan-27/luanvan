import sys
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
model = YOLO(str(ROOT / "models" / "drone.pt"))

r = model.predict(sys.argv[1], imgsz=640, conf=0.25, verbose=False, device="cpu")[0]
if len(r.boxes) == 0:
    print("no detections")
for b in r.boxes:
    print("class_id:", int(b.cls), "conf:", round(float(b.conf), 3),
          "box:", [round(v) for v in b.xyxy[0].tolist()])