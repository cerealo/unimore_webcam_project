import cv2
from ultralytics import SAM 
#from ultralytics.models.sam import SamAutomaticMaskGenerator

print("inserisci le coordinate dell'area degli oggetti che vuoi selezionare: ", end=" ")

x1=input()
print(",", end=" ")
y1=input()
print(",", end=" ")
x2=input()
print(",", end=" ")
y2=input()

#model = SAM("sam2_b.pt")
model = SAM("mobile_sam.pt")

model.eval()

model.info()

results = model(source="psyduck2.jpg", bboxes=[int(x1), int(y1), int(x2), int(y2)], show=True, save=True)

print(len(results[0].masks))