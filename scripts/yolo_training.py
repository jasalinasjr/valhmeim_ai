from ultralytics import YOLO

# Use YOLOv11 nano - best balance for your GPU
model = YOLO("yolov11n.pt")

results = model.train(
    data="valheim_dataset/data.yaml",
    epochs=80,                    # 50-80 is good starting point
    imgsz=640,
    batch=8,                      # Safe for 6GB VRAM on 970M
    device=0,
    name="valheim_yolo_v11",
    patience=15,                  # Early stopping
    pretrained=True,
    optimizer="AdamW"
)
