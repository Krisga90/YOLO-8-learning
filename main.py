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
results = model2.predict(source='data/video1.avi',  save=False, conf=0.5, save_txt=False)
# print(a.boxes)

lista_list = [[]]
for det in results:

    boxes = det.boxes
    for box in boxes:
        lista = []
        for data in box.data[0]:
            lista.append(data.item())
        lista_list.append(lista)

print(lista_list)




