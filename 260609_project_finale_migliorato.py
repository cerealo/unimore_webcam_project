import cv2
import numpy as np
from ultralytics import SAM

# =========================================================
# PARAMETRI
# =========================================================

# Immagine da analizzare

#IMAGE_PATH = "psyduck.jpg"
#IMAGE_PATH = "psyduck2.jpg"
#IMAGE_PATH = "psycacc.jpg"
IMAGE_PATH = "ifigos.jpg"
#IMAGE_PATH = "psyduckbello.jpg"
#IMAGE_PATH = "nonpsyduck.jpg"

# Numero di cluster per KMeans
K = 5

# Area minima per considerare valido un oggetto
MIN_AREA = 800


# =========================================================
# COLORE HSV -> NOME COLORE
# =========================================================

def get_color_name(h, s, v):
    # Nero
    if v < 40:
        return "Black"

    # Bianco
    if s < 20 and v > 210:
        return "White"

    # Grigio
    if s < 20:
        return "Gray"

    # Mappatura Hue -> colore
    if 0 <= h < 8 or 172 <= h <= 180:
        return "Red"

    if 8 <= h < 18:
        return "Orange"

    if 18 <= h < 28:
        return "Yellow"

    if 28 <= h < 70:
        return "Green"

    if 70 <= h < 105:
        return "Cyan"

    if 105 <= h < 130:
        return "Blue"

    if 130 <= h < 155:
        return "Purple"

    if 155 <= h < 172:
        return "Pink"

    return "Unknown"

# =========================================================
# PIPELINE COMPLETA FILTRAGGIO BBOX
# =========================================================

# =========================================================
# BBOX FILTERS
# =========================================================

def filter_border_bboxes(boxes, img_shape, margin=3):

    h, w = img_shape[:2]
    out = []

    for (x, y, bw, bh) in boxes:

        if x <= margin:
            continue

        if y <= margin:
            continue

        if x + bw >= w - margin:
            continue

        if y + bh >= h - margin:
            continue

        out.append((x, y, bw, bh))

    return out


def filter_contained_bboxes(boxes, contain_thr=0.90):

    kept = []

    for i, a in enumerate(boxes):

        ax, ay, aw, ah = a
        area_a = aw * ah

        contained = False

        for j, b in enumerate(boxes):

            if i == j:
                continue

            bx, by, bw, bh = b

            ix1 = max(ax, bx)
            iy1 = max(ay, by)
            ix2 = min(ax + aw, bx + bw)
            iy2 = min(ay + ah, by + bh)

            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)

            inter = iw * ih

            if inter / float(area_a + 1e-6) > contain_thr:

                if (bw * bh) > area_a:
                    contained = True
                    break

        if not contained:
            kept.append(a)

    return kept

# =========================================================
# IOU (INTERSECTION OVER UNION)
# =========================================================
#
# Misura quanto due bbox si sovrappongono.
#
# 0 = nessuna sovrapposizione
# 1 = identiche
#
# =========================================================

def iou(a, b):

    xA = max(a[0], b[0])
    yA = max(a[1], b[1])

    xB = min(a[0] + a[2], b[0] + b[2])
    yB = min(a[1] + a[3], b[1] + b[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)

    inter = interW * interH

    if inter == 0:
        return 0

    areaA = a[2] * a[3]
    areaB = b[2] * b[3]

    return inter / float(areaA + areaB - inter + 1e-6)


# =========================================================
# RIMOZIONE BBOX DUPLICATE
# =========================================================
#
# Se due bbox si sovrappongono troppo,
# tiene solo quella più grande.
#
# =========================================================



def filter_overlaps(boxes, thr=0.4):

    boxes = sorted(
        boxes,
        key=lambda b: b[2] * b[3],
        reverse=True
    )

    kept = []

    for b in boxes:

        keep = True

        for k in kept:

            if iou(b, k) > thr:
                keep = False
                break

        if keep:
            kept.append(b)

    return kept


def clean_bboxes(boxes, img_shape):

    boxes = filter_border_bboxes(
        boxes,
        img_shape
    )

    boxes = filter_contained_bboxes(
        boxes
    )

    boxes = filter_overlaps(
        boxes
    )

    return boxes

def merge_points(points, dist_thr=35):

    merged = []

    for p in points:

        keep = True

        for q in merged:

            d = np.hypot(
                p[0] - q[0],
                p[1] - q[1]
            )

            if d < dist_thr:
                keep = False
                break

        if keep:
            merged.append(p)

    return merged

# =========================================================
# INVERTI COLORI DELLE BBOX
# =========================================================

def invert_bbox_colors(image, boxes, output_path="psyduck_inverted.png"):

    inverted_img = image.copy()

    for (x, y, w, h) in boxes:

        roi = inverted_img[y:y+h, x:x+w]

        inverted_img[y:y+h, x:x+w] = 255 - roi

    cv2.imwrite(output_path, inverted_img)

    return inverted_img

# =========================================================
# CARICAMENTO IMMAGINE
# =========================================================

img = cv2.imread(IMAGE_PATH)

# ridimensionamento fisso
img = cv2.resize(img, (500, 500))

# HSV per colori
hsv = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2HSV
)

# grayscale per edge detection
gray = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)

# =========================================================
# KMEANS
# =========================================================
#
# Trova regioni colore dominanti.
#
# =========================================================

pixels = img.reshape((-1, 3)).astype(np.float32)

_, labels, _ = cv2.kmeans(
    pixels,
    K,
    None,
    (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        20,
        1.0
    ),
    10,
    cv2.KMEANS_PP_CENTERS
)

labels = labels.flatten()

# centroidi trovati tramite colore
color_centroids = []

for i in range(K):

    mask = (
        (labels == i)
        .astype(np.uint8)
        .reshape((500, 500))
        * 255
    )

    # pulizia rumore
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((5, 5), np.uint8)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8)
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for c in contours:

        if cv2.contourArea(c) < MIN_AREA:
            continue

        M = cv2.moments(c)

        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        color_centroids.append(
            (cx, cy)
        )

# =========================================================
# SAM
# =========================================================
#
# Usa tutti i centroidi trovati
# come prompt per segmentare oggetti.
#
# =========================================================

model = SAM("mobile_sam.pt")

output = img.copy()

print("Prima:", len(color_centroids))

all_points = merge_points(color_centroids, 35)

print("Dopo:", len(all_points))

detections = []

for (x, y) in all_points:

    results = model(
        img,
        points=[[x, y]],
        labels=[1]
    )

    if results[0].masks is None:
        continue

    masks = results[0].masks.data.cpu().numpy()

    for mask in masks:

        mask = (mask > 0.5).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for c in contours:

            if cv2.contourArea(c) < MIN_AREA:
                continue

            M = cv2.moments(c)

            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            x1, y1, w, h = cv2.boundingRect(c)

            detections.append({
                "bbox": (x1, y1, w, h),
                "contour": c,
                "center": (cx, cy)
            })

# =========================================================
# FILTER BBOX
# =========================================================
#
# Qui applichi la pipeline di pulizia:
# - rimuove bbox ai bordi
# - rimuove bbox contenute
# - rimuove bbox sovrapposte
#
# =========================================================

boxes = [d["bbox"] for d in detections]

filtered_boxes = clean_bboxes(boxes, output.shape)


# =========================================================
# DRAW ONLY VALID DETECTIONS
# =========================================================
#
# Disegna SOLO le detection che hanno passato i filtri
#
# =========================================================

for d in detections:

    # se bbox eliminata dai filtri → skip
    if d["bbox"] not in filtered_boxes:
        continue

    x, y, w, h = d["bbox"]

    # contorno oggetto segmentato da SAM
    cv2.drawContours(output, [d["contour"]], -1, (0,255,255), 2)
    # centro di massa del contorno
    cv2.circle(output, d["center"], 5, (0,0,255), -1)
    # bounding box finale pulita
    cv2.rectangle(output, (x,y), (x+w, y+h), (255,0,0), 2)

# =========================================================
# OUTPUT FINALE
# =========================================================
#
# Mostra risultato finale:
# - oggetti segmentati
# - bbox filtrate
# - centroidi puliti
#
# =========================================================

inverted_output = invert_bbox_colors(
    img,
    filtered_boxes,
    "psyduck_inverted.png"
)

cv2.imshow("INVERTED BOXES", inverted_output)
cv2.imshow("IMAGE", output)
cv2.waitKey(0)
cv2.destroyAllWindows()