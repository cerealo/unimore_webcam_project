#implementazione della libreria cv2
import cv2

#img contiene l'immagine psyduck.jpg
img = cv2.imread("psyduck.jpg")

#mostra tramite una finesta l'immagine img dandole nome Immagine
cv2.imshow("Immagine", img)
#aspetta finche non viene premuto un tasto
cv2.waitKey(0)
#distrugge tutte le finestre
cv2.destroyAllWindows()

