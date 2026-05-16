import cv2
import threading
import time
from validaciones import AnalizadorHumano
from procesamiento_frame import crear_texto_procesamiento

class CapturadorVideoCamara:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.ret, self.frame = self.cap.read()
        self.running = True
        
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()

    def _update(self):
        while self.running:
            if self.cap.isOpened():
                self.ret, self.frame = self.cap.read()
            time.sleep(0.01) 

    def leer_frame(self):
        return self.ret, self.frame

    def liberar(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()


camara = CapturadorVideoCamara(src=0, width=640, height=480)


analizador = AnalizadorHumano()

print("Presiona 'q' para salir")

while True:
    
    ret, frame = camara.leer_frame()

    if not ret or frame is None:
        print("Esperando señal de la cámara...")
        continue

    color_verde = (0, 255, 0)
    color_rojo = (0, 0, 255)
    mensajes = []

    
    analizador.actualizar_frame(frame)

    
    hay_humano = analizador.detectar_humano()

    if hay_humano:
        mensajes.append("HUMANO DETECTADO")
        
        
        if analizador.detectar_cara():
            mensajes.append("CARA DETECTADA")

        if analizador.esta_de_frente():
            mensajes.append("DE FRENTE")
        elif analizador.esta_de_espalda():
            mensajes.append("DE ESPALDA")
        elif analizador.esta_de_lado_izquierdo():
            mensajes.append("LADO IZQUIERDO")
        elif analizador.esta_de_lado_derecho():
            mensajes.append("LADO DERECHO")

        if analizador.detectar_mano_izquierda():
            mensajes.append("MANO IZQUIERDA")
        if analizador.detectar_mano_derecha():
            mensajes.append("MANO DERECHA")

        if analizador.detectar_tronco():
            mensajes.append("TRONCO")
        if analizador.detectar_brazo_izquierdo():
            mensajes.append("BRAZO IZQUIERDO")
        if analizador.detectar_brazo_derecho():
            mensajes.append("BRAZO DERECHO")
        if analizador.detectar_pierna_izquierda():
            mensajes.append("PIERNA IZQUIERDA")
        if analizador.detectar_pierna_derecha():
            mensajes.append("PIERNA DERECHA")
    else:
        mensajes.append("NO SE DETECTA HUMANO")

    
    texto = "\n".join(mensajes)
    color = color_verde if hay_humano else color_rojo

    frame_procesado = crear_texto_procesamiento(frame, texto, color)

    cv2.imshow('Deteccion de Humanos', frame_procesado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara.liberar()
cv2.destroyAllWindows()