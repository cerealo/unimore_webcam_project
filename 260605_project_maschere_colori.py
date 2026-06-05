import cv2
import numpy as np

# =========================
# PARAMETRI
# =========================
IMAGE_PATH = "psyduck2.jpg"
K = 5  # numero di colori da estrarre

# =========================
# CARICA IMMAGINE
# =========================
img = cv2.imread(IMAGE_PATH)
img = cv2.resize(img, (500, 500))  # opzionale

# reshape in lista pixel
pixels = img.reshape((-1, 3))
pixels = np.float32(pixels)

# =========================
# K-MEANS
# =========================
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
_, labels, centers = cv2.kmeans(
    pixels,
    K,
    None,
    criteria,
    10,
    cv2.KMEANS_RANDOM_CENTERS
)

centers = np.uint8(centers)
labels = labels.flatten()

# =========================
# CREA MASCHERE
# =========================
masks = []

for i in range(K):
    mask = (labels == i).astype(np.uint8) * 255
    mask = mask.reshape(img.shape[:2])
    masks.append(mask)

# =========================
# MOSTRA RISULTATI
# =========================
for i, mask in enumerate(masks):
    # cv2.imshow(f"Mask {i}", mask)

    # opzionale: colore associato
    color = np.full_like(img, centers[i])
    result = cv2.bitwise_and(img, color, mask=mask)
    cv2.imshow(f"Color {i}", result)

cv2.imshow("Original", img)
cv2.waitKey(0)
cv2.destroyAllWindows()