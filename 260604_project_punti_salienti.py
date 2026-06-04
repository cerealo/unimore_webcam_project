import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ======================================================
    # 1) PRE-PROCESSING MIGLIORATO (meno rumore + meno ombre)
    # ======================================================
    gray_blur = cv2.GaussianBlur(gray, (11, 11), 0)

    # ======================================================
    # 2) SOGGLIA AUTOMATICA STABILE
    # ======================================================
    _, thresh = cv2.threshold(
        gray_blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ======================================================
    # 3) RIMOZIONE BLOB PICCOLI (OPEN più forte ma controllato)
    # ======================================================
    kernel_small = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small, iterations=2)

    # ======================================================
    # 4) TAPPA BUCCHI / UNISCE OGGETTI
    # ======================================================
    kernel_med = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_med, iterations=2)

    # ======================================================
    # 5) FILL HOLES (rendere oggetti solidi)
    # ======================================================
    contours_fill, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(thresh, contours_fill, -1, 255, thickness=cv2.FILLED)

    # ======================================================
    # DEBUG VIEW
    # ======================================================
    cv2.imshow("clean mask", thresh)

    # ======================================================
    # 6) CONTORNI FINALI
    # ======================================================
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = gray.shape

    for c in contours:
        area = cv2.contourArea(c)

        # elimina rumore residuo
        if area < 1000:
            continue

        # centro di massa
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # disegno contorno
        cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

        # centro
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

        cv2.putText(frame, "OBJ", (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("Live Multi Object (UPGRADED)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
