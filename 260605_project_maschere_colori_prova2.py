import cv2
import numpy as np

# =========================
# PARAMETRI
# =========================
IMAGE_PATH = "ifigos.jpg"
K = 7
MIN_AREA = 800

# =========================
# COLORE -> NOME (HSV)
# =========================
def get_color_name(h, s, v):

    # black / white / gray più separati
    if v < 40:
        return "Black"
    if s < 20 and v > 210:
        return "White"
    if s < 20:
        return "Gray"

    # 🔥 range più stretti = più differenza tra colori

    if 0 <= h < 8 or 172 <= h <= 180:
        return "Red"

    if 8 <= h < 18:
        return "Orange"

    if 18 <= h < 28:
        return "Yellow"

    if 28 <= h < 70:
        return "Green"

    if 70 <= h < 105:
        return "Cyan"   # 🔥 separato dal blu

    if 105 <= h < 130:
        return "Blue"

    if 130 <= h < 155:
        return "Purple"

    if 155 <= h < 172:
        return "Pink"

    return "Unknown"


# =========================
# CARICA IMMAGINE
# =========================
img = cv2.imread(IMAGE_PATH)

if img is None:
    print("Immagine non trovata")
    exit()

img = cv2.resize(img, (500, 500))

# =========================
# HSV VERSIONE
# =========================
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# =========================
# KMEANS (SEGMENTAZIONE OGGETTI)
# =========================
pixels = img.reshape((-1, 3)).astype(np.float32)

criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    20,
    1.0
)

_, labels, centers = cv2.kmeans(
    pixels,
    K,
    None,
    criteria,
    10,
    cv2.KMEANS_PP_CENTERS
)

labels = labels.flatten()

# =========================
# OUTPUT
# =========================
color_masks = {}

# =========================
# LOOP KMEANS
# =========================
for i in range(K):

    mask = (labels == i).astype(np.uint8)
    mask = mask.reshape(img.shape[:2])

    # pulizia rumore
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # riempi oggetti
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)

    # salta troppo piccoli
    if np.sum(filled) < MIN_AREA:
        continue

    # =========================
    # COLORE REALE DA HSV (FIX BLACK/WHITE)
    # =========================
    pixels_in = hsv[filled == 255]

    if len(pixels_in) == 0:
        continue

   # h, s, v = np.mean(pixels_in, axis=0)
   # usa mediana per separare meglio i colori
    h = np.median(pixels_in[:, 0])
    s = np.median(pixels_in[:, 1])
    v = np.median(pixels_in[:, 2])

    color_name = get_color_name(h, s, v)

    # =========================
    # MERGE COLORI
    # =========================
    if color_name not in color_masks:
        color_masks[color_name] = filled
    else:
        color_masks[color_name] = cv2.bitwise_or(color_masks[color_name], filled)

# =========================
# VISUALIZZAZIONE
# =========================
for name, mask in color_masks.items():
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow(name, result)

cv2.imshow("Original", img)

cv2.waitKey(0)
cv2.destroyAllWindows()