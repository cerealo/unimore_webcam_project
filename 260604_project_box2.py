import cv2
from ultralytics import SAM 

# Prendiamo i punti dall'utente
print("Inserisci le coordinate dell'area (x1, y1, x2, y2):")
x1 = int(input("x1: "))
y1 = int(input("y1: "))
x2 = int(input("x2: "))
y2 = int(input("y2: "))

# Carichiamo l'immagine originale con OpenCV
img = cv2.imread("psyduck2.jpg")

# Ritagliamo l'area di interesse usando lo slicing di NumPy [y1:y2, x1:x2]
cropped_img = img[y1:y2, x1:x2]

# Carichiamo il modello (MobileSAM o SAM2)
#model = SAM("mobile_sam.pt")
model = SAM("sam2_b.pt")

# Avviamo la predizione sull'immagine ritagliata *senza* bboxes.
# Questo attiverà la modalità automatica su tutta l'area ritagliata.
results = model.predict(source=cropped_img, show=True, save=True)

# Adesso vedrai che la lista conterrà molte più maschere
if results[0].masks is not None:
    print(f"Oggetti trovati nell'area: {len(results[0].masks)}")
else:
    print("Nessun oggetto trovato.")