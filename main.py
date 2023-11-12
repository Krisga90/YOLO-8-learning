from IPython.display import Image
import torch
from ultralytics import YOLO

print(torch.cuda.is_available())
print(torch.__version__)

# initialize YOLO with the model name
model = YOLO("yolov8s.pt")
model.predict(source="./data/im1.jpg", save=True, conf=0.5, save_txt=True)

Image("runs/detect/predict/labels/im1.jpg")


model2 = YOLO("yolov8m.pt")
model2.predict(source='data/video1.avi',  save=True, conf=0.5, save_txt=False)

