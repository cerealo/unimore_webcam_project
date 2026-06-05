import cv2
import numpy as np

def contorna_oggetti_colorati(immagine_sorgente, x1, y1, x2, y2):
    """
    Analizza la zona selezionata, contorna di giallo l'oggetto,
    trova il centro di massa e inserisce un PALLINO ROSSO gigante nel baricentro.
    """
    immagine_output = immagine_sorgente.copy()
    altezza, larghezza, _ = immagine_sorgente.shape
    
    # 1. Validazione e ordinamento delle coordinate della zona di ricerca
    x1 = max(0, min(x1, larghezza - 1))
    x2 = max(0, min(x2, larghezza - 1))
    y1 = max(0, min(y1, altezza - 1))
    y2 = max(0, min(y2, altezza - 1))

    start_x, end_x = min(x1, x2), max(x1, x2)
    start_y, end_y = min(y1, y2), max(y1, y2)

    # Ritaglio della ROI
    zona_ritagliata = immagine_sorgente[start_y:end_y, start_x:end_x]
    
    if zona_ritagliata.size == 0:
        print("Zona non valida!")
        return immagine_output

    # 2. Conversione in HSV
    hsv_zona = cv2.cvtColor(zona_ritagliata, cv2.COLOR_BGR2HSV)

    # 3. Filtro colore (Saturazione minima a 40 per scartare lo sfondo grigio/nero)
    limite_inferiore_colore = np.array([0, 40, 40])
    limite_superiore_colore = np.array([180, 255, 255])

    # Creiamo la maschera
    maschera_colori = cv2.inRange(hsv_zona, limite_inferiore_colore, limite_superiore_colore)

    # 4. Pulizia del rumore di fondo
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    maschera_pulita = cv2.morphologyEx(maschera_colori, cv2.MORPH_OPEN, kernel)

    # 5. Trova i contorni delle forme isolate
    contorni, _ = cv2.findContours(maschera_pulita, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area_minima_pixel = 150 
    lista_centri = []

    for c in contorni:
        area = cv2.contourArea(c)
        if area > area_minima_pixel:
            
            # Trasliamo il contorno dalla ROI all'immagine intera per il disegno
            contorno_traslato = c + np.array([start_x, start_y])
            
            # Disegniamo prima il bordo in GIALLO con spessore 3
            cv2.drawContours(immagine_output, [contorno_traslato], -1, (0, 255, 255), 3)
            
            # --- CALCOLO CENTRO DI MASSA ---
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX_local = int(M["m10"] / M["m00"])
                cY_local = int(M["m01"] / M["m00"])
                
                cX_global = cX_local + start_x
                cY_global = cY_local + start_y
                
                # Salviamo il centro per disegnarlo alla fine sopra i contorni
                lista_centri.append((cX_global, cY_global))

    # 6. DISEGNO DEI PALLINI ROSSI (Fatto alla fine per sovrascrivere tutto ed essere visibile)
    for centro in lista_centri:
        # Disegnamo un pallino ROSSO acceso (BGR: 0, 0, 255), raggio aumentato a 8 pixel, pieno (-1)
        cv2.circle(immagine_output, centro, 8, (0, 0, 255), -1)
        
        # Aggiungiamo anche un cerchio di contorno nero attorno al pallino rosso per farlo risaltare
        cv2.circle(immagine_output, centro, 8, (0, 0, 0), 2)
        
        # Scriviamo le coordinate sopra il pallino
        testo = f"X:{centro[0]}, Y:{centro[1]}"
        cv2.putText(immagine_output, testo, (centro[0] + 15, centro[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(immagine_output, testo, (centro[0] + 15, centro[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return immagine_output

# --- ESECUZIONE ---
if __name__ == "__main__":
    img = cv2.imread("psyduck2.jpg")
    
    if img is None:
        print("Errore: Immagine non trovata.")
        exit()

    # Coordinate del tuo test (Modificale a piacimento)
    box_x1, box_y1 = 400, 200
    box_x2, box_y2 = 300, 294

    # Elaborazione
    risultato_visivo = contorna_oggetti_colorati(img, box_x1, box_y1, box_x2, box_y2)

    # Disegniamo la tua box di ricerca in ROSSO
    cv2.rectangle(risultato_visivo, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 255), 2)
    
    cv2.imshow("Centro di Massa e Contorni", risultato_visivo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()