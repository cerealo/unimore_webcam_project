import cv2
import numpy as np
from ultralytics import SAM

# =========================================================
# PARAMETRI
# =========================================================

# Immagine da analizzare
IMAGE_PATH = "psyduck.jpg"

# Numero di cluster per KMeans
K = 7

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
# FILTRO BOUNDING BOX SUI BORDI
# =========================================================
#
# Elimina bounding box che toccano
# i bordi dell'immagine.
#
# Utile per rimuovere:
# - sfondo
# - frame
# - artefatti laterali
#
# =========================================================

def filter_border_bboxes(boxes, img_shape, margin=3):

    h, w = img_shape[:2]

    out = []

    for (x, y, w1, h1) in boxes:

        # bordo sinistro o superiore
        if x <= margin or y <= margin:
            continue

        # bordo destro o inferiore
        if (x + w1) >= (w - margin):
            continue

        if (y + h1) >= (h - margin):
            continue

        out.append((x, y, w1, h1))

    return out


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


# =========================================================
# PIPELINE COMPLETA FILTRAGGIO BBOX
# =========================================================

def clean_bboxes(boxes, img_shape):

    # filtro bordi
    boxes = filter_border_bboxes(
        boxes,
        img_shape
    )

    # filtro duplicati
    boxes = filter_overlaps(
        boxes
    )

    return boxes

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
# PSYDUCK DETECTOR
# =========================================================
#
# Usa:
# - Canny
# - Connected Components
#
# per trovare centroidi iniziali.
#
# =========================================================

edges = cv2.Canny(
    gray,
    70,
    170
)

# chiusura dei bordi
edges = cv2.morphologyEx(
    edges,
    cv2.MORPH_CLOSE,
    np.ones((9, 9), np.uint8)
)

# ispessimento edge
edges = cv2.dilate(
    edges,
    np.ones((5, 5), np.uint8),
    iterations=1
)

# componenti connesse
num, cc = cv2.connectedComponents(edges)

# punti candidati Psyduck
psyduck_centroids = []

for i in range(1, num):

    comp = (cc == i).astype(np.uint8) * 255

    area = cv2.countNonZero(comp)

    # scarta oggetti piccoli
    if area < MIN_AREA:
        continue

    x, y, w, h = cv2.boundingRect(comp)

    # scarta blob minuscoli
    if w < 25 or h < 25:
        continue

    # scarta blob enormi
    if w * h > 0.75 * 500 * 500:
        continue

    # riempimento componente
    contours, _ = cv2.findContours(
        comp,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filled = np.zeros_like(comp)

    cv2.drawContours(
        filled,
        contours,
        -1,
        255,
        cv2.FILLED
    )

    ys, xs = np.where(filled > 0)

    if len(xs) > 0:

        cx = int(np.mean(xs))
        cy = int(np.mean(ys))

        psyduck_centroids.append(
            (cx, cy)
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

all_points = (
    psyduck_centroids +
    color_centroids
)

detections = []

for (x, y) in all_points:

    results = model(img, points=[[x, y]], labels=[1])

    masks = results[0].masks.data.cpu().numpy()

    if len(masks) == 0:
        continue

    mask = (masks[0] > 0.5).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:

        if cv2.contourArea(c) < MIN_AREA:
            continue

        M = cv2.moments(c)

        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            continue

        x1 = xs.min()
        y1 = ys.min()
        x2 = xs.max()
        y2 = ys.max()

        detections.append({
            "bbox": (x1, y1, x2 - x1, y2 - y1),
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
# SAM
# =========================================================
#
# Usa tutti i centroidi trovati (Psyduck + KMeans)
# come prompt per segmentare oggetti con SAM.
#
# =========================================================

model = SAM("mobile_sam.pt")

# immagine di output su cui disegnare risultati finali
output = img.copy()

# unione di tutti i punti candidati
all_points = (
    psyduck_centroids +
    color_centroids
)

# lista finale delle detection (bbox + contorno + centro)
detections = []

for (x, y) in all_points:

    # inferenza SAM con punto guida
    results = model(img, points=[[x, y]], labels=[1])

    # estrazione maschere (tensor -> numpy)
    masks = results[0].masks.data.cpu().numpy()

    # se non trova maschere salta
    if len(masks) == 0:
        continue

    # binarizzazione maschera principale
    mask = (masks[0] > 0.5).astype(np.uint8) * 255

    # pulizia rumore (piccoli buchi o pixel isolati)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    # estrazione contorni dalla maschera
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for c in contours:

        # scarta oggetti troppo piccoli
        if cv2.contourArea(c) < MIN_AREA:
            continue

        # centro di massa del contorno
        M = cv2.moments(c)

        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # coordinate pixel attivi della maschera
        ys, xs = np.where(mask > 0)

        # sicurezza: evita maschere vuote
        if len(xs) == 0 or len(ys) == 0:
            continue

        # bounding box dalla maschera
        x1 = xs.min()
        y1 = ys.min()
        x2 = xs.max()
        y2 = ys.max()

        # salva detection completa
        detections.append({
            "bbox": (x1, y1, x2 - x1, y2 - y1),
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
    cv2.drawContours(
        output,
        [d["contour"]],
        -1,
        (0,255,255),
        2
    )

    # centro di massa del contorno
    cv2.circle(
        output,
        d["center"],
        5,
        (0,0,255),
        -1
    )

    # bounding box finale pulita
    cv2.rectangle(
        output,
        (x,y),
        (x+w, y+h),
        (255,0,0),
        2
    )


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

cv2.imshow("PSYDUCK FIXED + SAM", output)
cv2.waitKey(0)
cv2.destroyAllWindows()