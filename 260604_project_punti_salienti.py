import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ✔ riduce ombre e rumore
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # ✔ soglia automatica
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ✔ pulizia (il tuo kernel integrato correttamente)
    kernel = np.ones((9, 9), np.uint8)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # trova contorni
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for c in contours:
        area = cv2.contourArea(c)

        # elimina rumore
        if area < 800:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # disegna contorno
        cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

        # centro di massa
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

        # etichetta
        cv2.putText(frame, "OBJ", (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("Live Multi Object + Shadows Reduced", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
