import cv2

def crear_texto_procesamiento(frame, texto, color):
    y = 30
    for linea in texto.split("\n"):
        cv2.putText(
            frame,
            linea,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        y += 30
    return frame