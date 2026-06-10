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

# Area minima per considerare valido un oggetto
MIN_AREA = 400

"""Rimuove i punti troppo vicini dividendo lo spazio in una griglia e tenendo un solo punto per cella."""
def spatial_deduplicate(points, cell_size=80): 
    grid = set()
    out = []
    for x, y in points:
        cell = (x // cell_size, y // cell_size)
        if cell in grid:
            continue
        grid.add(cell)
        out.append((x, y))
    return out


"""Genera una griglia regolare di punti distanziati in base alle dimensioni dell'immagine."""
def generate_grid_points(img_shape, step=60):
    h, w = img_shape[:2]
    pts = []
    for y in range(step, h, step):
        for x in range(step, w, step):
            pts.append((x, y))
    return pts


"""Rimuove i bounding box che toccano o sono troppo vicini ai bordi dell'immagine."""
def filter_border_bboxes(boxes, img_shape, margin=3):
    h, w = img_shape[:2]
    out = []
    for x, y, bw, bh in boxes:
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


"""Elimina i bounding box che sono quasi completamente contenuti dentro un altro box più grande."""
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
            # Calcola l'intersezione tra il box A e il box B
            ix1 = max(ax, bx)
            iy1 = max(ay, by)
            ix2 = min(ax + aw, bx + bw)
            iy2 = min(ay + ah, by + bh)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            # Se A è dentro B ed è più piccolo, viene marcato come contenuto
            if inter / float(area_a + 1e-6) > contain_thr:
                if (bw * bh) > area_a:
                    contained = True
                    break
        if not contained:
            kept.append(a)
    return kept


"""Calcola l'Intersection over Union (IoU), ovvero la percentuale di sovrapposizione tra due box."""
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


"""Ordina i box per area e rimuove quelli che si sovrappongono troppo (IoU > thr) a box più grandi."""
def filter_overlaps(boxes, thr=0.4):
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
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


"""Applica in sequenza i tre filtri di pulizia (bordi, box contenuti e sovrapposizioni)."""
def clean_bboxes(boxes, img_shape):
    boxes = filter_border_bboxes(boxes, img_shape)
    boxes = filter_contained_bboxes(boxes)
    boxes = filter_overlaps(boxes)
    return boxes


"""Rimuove i punti che si trovano a una distanza geometrica inferiore alla soglia impostata."""
def merge_points(points, dist_thr=35):
    merged = []
    for p in points:
        keep = True
        for q in merged:
            d = np.hypot(p[0] - q[0], p[1] - q[1])
            if d < dist_thr:
                keep = False
                break
        if keep:
            merged.append(p)
    return merged


"""Inverte i colori (effetto negativo) solo nelle aree racchiuse dai bounding box e salva l'immagine."""
def invert_bbox_colors(image, boxes, output_path="psyduck_inverted.png"):
    inverted_img = image.copy()
    for x, y, w, h in boxes:
        roi = inverted_img[y : y + h, x : x + w]
        inverted_img[y : y + h, x : x + w] = 255 - roi
    cv2.imwrite(output_path, inverted_img)
    return inverted_img

# MAIN

img = cv2.imread(IMAGE_PATH)

# ridimensionamento fisso
img = cv2.resize(img, (500, 500))

# grayscale per edge detection
gray = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)

# =========================================================
# PSYDUCK DETECTOR
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
# SAM
# 
# Usa tutti i centroidi trovati
# come prompt per segmentare oggetti.
#
# =========================================================

model = SAM("mobile_sam.pt")

output = img.copy()

grid_points = generate_grid_points(img.shape, step=60)

all_points = merge_points(
    psyduck_centroids + grid_points,
    dist_thr=30
)

all_points = spatial_deduplicate(all_points, cell_size=80)

detections = []

for (x, y) in all_points:

    results = model(
    img,
    points=[[x, y]],
    labels=[1],
)

    masks = results[0].masks.data.cpu().numpy()

    if len(masks) == 0:
        continue

    masks = results[0].masks.data.cpu().numpy()

    # scegli la mask più grande (più stabile)
    areas = [m.sum() for m in masks]
    best = np.argmax(areas)

    mask = (masks[best] > 0.5).astype(np.uint8) * 255


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
cv2.imshow("IMAGE FINAL", output)
cv2.waitKey(0)
cv2.destroyAllWindows()