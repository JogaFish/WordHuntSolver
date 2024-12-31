import cv2
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
results = model.train(data="config.yaml", epochs=10, pretrained=True)

results = model.predict(source=0, show=True)


