import cv2
from ultralytics import SAM 

model = SAM("mobile_sam.pt")

model.eval()

model.info()

results = model(source="psyduck.jpg", points=([400, 200], [300, 250], [100, 100]), show=True, save=True)

#results = model(source="psyduck2.jpg", show=True, save=True)

print(results)
print(len(results[0].masks))