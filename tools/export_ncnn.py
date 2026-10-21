from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]   # = ...\ObjectDetectCore
WEIGHTS = ROOT / "models" / "drone.pt"

model = YOLO(str(WEIGHTS))
print("Class names:", model.names)

out_dir = Path(model.export(format="ncnn", imgsz=640, batch=1,
                            half=False, device="cpu"))
print("Exported to:", out_dir)

names = [model.names[i] for i in sorted(model.names)]
(out_dir / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

(out_dir / "detectcore.cfg").write_text(
    "input=in0\noutput=out0\nwidth=640\nheight=640\nlayout=raw_xywh\n",
    encoding="utf-8",
)
print("Files:", sorted(p.name for p in out_dir.iterdir()))