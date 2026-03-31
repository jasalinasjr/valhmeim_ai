from ultralytics import YOLO

model = YOLO("yolov8n.pt")        # or "yolov8s.pt"

results = model.train(
    data="path/to/your_dataset/data.yaml",
    epochs=80,                    # start with 50–100
    imgsz=640,                    # 512 if VRAM is tight
    batch=8,                      # 4–8 max on 970M
    workers=2,                    # low to save RAM
    device=0,                     # GPU
    name="valheim_yolo_v1",
    patience=15,                  # early stopping
    pretrained=True,
    optimizer="AdamW",
    lr0=0.01,                     # learning rate
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005
)
