#implementazione della libreria openCV
import cv2

#specifica della webcam in uso (0 quella di default, 1,2,3,ecc per quelle aggiuntive)
cam = cv2.VideoCapture(1)

#ciclo infinito
while True:

    #frame legge l'input di cam ovvero la webcam
    ret, frame = cam.read()

    #viene mostrata la webcam in una finesta chiamata Webcam
    cv2.imshow("Webcam", frame)

    #se esc viene premuto il programma viene interrotto
    if cv2.waitKey(1) == 27:  # ESC
        break

#libera la webcam
cam.release()
#distrugge tutte le finestre aperte dal programma
cv2.destroyAllWindows()

