import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import FastSAM

# Costanti globali
MIN_AREA = 800

class NodeB(Node):
    def __init__(self):
        super().__init__('node_b')
        self.sub = self.create_subscription(Image, '/image_raw', self.listener_callback, 10)
        self.bridge = CvBridge()
        
        # Inizializza il modello FastSAM una sola volta all'avvio del nodo
        self.get_logger().info("Caricamento del modello FastSAM...")
        self.model = FastSAM("FastSAM-s.pt")
        
        self.frame_count = 0
        self.last_results = None
        self.get_logger().info("Nodo B pronto. In attesa di immagini da elaborare...")

    def listener_callback(self, msg):
        try:
            # 1. Riceve e converte l'immagine inviata dal Nodo A
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 2. Esegue la funzione di elaborazione (ex-main)
            self.process_image(cv_image)
            
            # 3. Aggiorna l'interfaccia grafica
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Errore durante la ricezione o elaborazione: {e}")

    def process_image(self, img):
        """Funzione principale che racchiude tutta la logica di elaborazione e rilevamento."""
        img = cv2.resize(img, (512, 512))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- PSYDUCK DETECTOR (Canny + Connected Components) ---
        edges = cv2.Canny(gray, 70, 170)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

        cv2.imshow("Edges (Canny) 1", edges)

        num, cc = cv2.connectedComponents(edges)
        psyduck_centroids = []

        for i in range(1, num):
            comp = (cc == i).astype(np.uint8) * 255
            area = cv2.countNonZero(comp)

            if area < MIN_AREA: continue
            x, y, w, h = cv2.boundingRect(comp)
            if w < 25 or h < 25: continue
            if w * h > 0.75 * 500 * 500: continue

            contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(comp)
            cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)

            ys, xs = np.where(filled > 0)
            if len(xs) > 0:
                cx = int(np.mean(xs))
                cy = int(np.mean(ys))
                psyduck_centroids.append((cx, cy))

        debug = img.copy()
        for cx, cy in psyduck_centroids:
            cv2.circle(debug, (cx, cy), 6, (0, 0, 255), -1)
        cv2.imshow("Psyduck Centroids 2", debug)

        # --- FAST SAM SEGMENTATION ---
        self.frame_count += 1
        if self.frame_count % 5 == 0 or self.last_results is None:
            self.last_results = self.model(img, iou=0.9, retina_masks=False, imgsz=512)

        results = self.last_results
        masks = results[0].masks.data.cpu().numpy()
        
        frame_page = img.copy()
        for mask in masks: 
            color = np.random.randint(0, 255, (3), dtype=np.uint8)
            frame_page[mask > 0] = frame_page[mask > 0] * 0.5 + color * 0.5
        cv2.imshow("pagina", frame_page)

        detections = []
        for m in masks:
            mask = (m > 0.5).astype(np.uint8) * 255
            if cv2.countNonZero(mask) < MIN_AREA: continue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                if cv2.contourArea(c) < MIN_AREA: continue
                M = cv2.moments(c)
                if M["m00"] == 0: continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                x, y, w, h = cv2.boundingRect(c)
                detections.append({"bbox": (x, y, w, h), "contour": c, "center": (cx, cy)})

        if len(masks) > 0:
            areas = [m.sum() for m in masks]
            best = np.argmax(areas)
            mask = (masks[best] > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                if cv2.contourArea(c) < MIN_AREA: continue
                M = cv2.moments(c)
                if M["m00"] == 0: continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                ys, xs = np.where(mask > 0)
                if len(xs) == 0 or len(ys) == 0: continue
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                detections.append({"bbox": (x1, y1, x2 - x1, y2 - y1), "contour": c, "center": (cx, cy)})

        debug_sam = img.copy()
        for d in detections:
            cv2.drawContours(debug_sam, [d["contour"]], -1, (0, 255, 255), 2)
            cv2.circle(debug_sam, d["center"], 5, (0, 0, 255), -1)
        cv2.imshow("SAM Raw Detections 3", debug_sam)

        # --- BBOX FILTERING & CLEANING ---
        boxes = [d["bbox"] for d in detections]
        filtered_boxes = clean_bboxes(boxes, img.shape)

        debug_filtered = img.copy()
        for x, y, w, h in filtered_boxes:
            cv2.rectangle(debug_filtered, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("Filtered BBoxes 4", debug_filtered)

        # --- OUTPUT GENERATION ---
        output = img.copy()
        for d in detections:
            if d["bbox"] not in filtered_boxes: continue
            x, y, w, h = d["bbox"]
            mask_vis = np.zeros(output.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask_vis, [d["contour"]], -1, 255, cv2.FILLED)
            output[mask_vis > 0] = (0, 255, 255)
            cv2.circle(output, d["center"], 5, (0, 0, 255), -1)
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

        inverted_output = invert_bbox_colors(img, filtered_boxes)
        cv2.imshow("INVERTED BOXES 5", inverted_output)
        cv2.imshow("IMAGE FINAL 6", output)


# =========================================================
# FUNZIONI DI SUPPORTO PIPELINE (Fuori dalla classe)
# =========================================================

def filter_border_bboxes(boxes, img_shape, margin=3):
    h, w = img_shape[:2]
    out = []
    for x, y, bw, bh in boxes:
        if x <= margin or y <= margin or x + bw >= w - margin or y + bh >= h - margin: continue
        out.append((x, y, bw, bh))
    return out

def filter_contained_bboxes(boxes, contain_thr=0.90):
    kept = []
    for i, a in enumerate(boxes):
        ax, ay, aw, ah = a
        area_a = aw * ah
        contained = False
        for j, b in enumerate(boxes):
            if i == j: continue
            bx, by, bw, bh = b
            ix1, iy1 = max(ax, bx), max(ay, by)
            ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            if inter / float(area_a + 1e-6) > contain_thr and (bw * bh) > area_a:
                contained = True
                break
        if not contained: kept.append(a)
    return kept

def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: return 0
    return inter / float(a[2] * a[3] + b[2] * b[3] - inter + 1e-6)

def filter_overlaps(boxes, thr=0.2):
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for b in boxes:
        keep = True
        for k in kept:
            if iou(b, k) > thr:
                keep = False
                break
        if keep: kept.append(b)
    return kept

def clean_bboxes(boxes, img_shape):
    boxes = filter_border_bboxes(boxes, img_shape)
    boxes = filter_contained_bboxes(boxes)
    boxes = filter_overlaps(boxes)
    return boxes

def invert_bbox_colors(image, boxes):
    inverted_img = image.copy()
    for x, y, w, h in boxes:
        roi = inverted_img[y : y + h, x : x + w]
        inverted_img[y : y + h, x : x + w] = 255 - roi
    return inverted_img


# --- MAIN ROS2 ---
def main():
    rclpy.init()
    node = NodeB()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()