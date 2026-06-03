import cv2
from ultralytics import SAM 

model = SAM("sam2_b.pt")

model.eval()

model.info()

#model(source="psyduck.jpg", show=True)

results = model(source="psyduck.jpg", points=[400, 200], show=True, save=True)

print(results)