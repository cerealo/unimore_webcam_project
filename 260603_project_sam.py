import cv2
from ultralytics import SAM 

model = SAM("sam2_b.pt")

model.eval()

model.info()

#results = model(source="psyduck2.jpg", points=[400, 200], show=True, save=True)

results = model(source="psyduck2.jpg", show=True, save=True)

print(results)
print(len(results[0].masks))
