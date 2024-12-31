from ultralytics import YOLO

model = YOLO("yolov8s.pt")
results = model.train(data="config.yaml", epochs=10, pretrained=True)

model.save("predict_model.pt")
