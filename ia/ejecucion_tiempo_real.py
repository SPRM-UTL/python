# =====================================================================
# 🛠️ PARCHE DE COMPATIBILIDAD GOOGLE PROTOBUF / TENSORFLOW / MEDIAPIPE
# =====================================================================
import sys
import types

# 1. Evita el "ImportError: cannot import name 'runtime_version'" de TensorFlow
try:
    import google.protobuf.runtime_version as rv
except ImportError:
    mod = types.ModuleType('google.protobuf.runtime_version')
    mod.runtime_version = lambda *args, **kwargs: None
    mod.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None
    sys.modules['google.protobuf.runtime_version'] = mod

# 2. Evita el "AttributeError: ... FieldDescriptor object has no attribute 'label'" de MediaPipe
try:
    from google._upb._message import FieldDescriptor
    if not hasattr(FieldDescriptor, 'label'):
        FieldDescriptor.label = property(lambda self: self.mode)
except ImportError:
    pass
# =====================================================================

import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from collections import deque
from tensorflow.keras.models import load_model

# --- CONFIGURACIÓN ---
ACCIONES = ["aplaudir", "saludar", "reposo"]
LARGO_SECUENCIA = 30

# 1. Clase para leer la cámara en un hilo independiente (Tu versión ultra-eficiente)
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

# 2. Inicializar componentes principales
print("🧠 Cargando modelo de Inteligencia Artificial...")
modelo = load_model('detector_movimientos.h5')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.6, min_tracking_confidence=0.6)

camara = CapturadorVideoCamara(src=0, width=640, height=480)

# Buffer circular para mantener los últimos 30 frames de landmarks
buffer_frames = deque(maxlen=LARGO_SECUENCIA)

# Variables para controlar la lógica de activación
escaneo_activo = False
contador_confirmacion = 0
CONFIRMACION_UMBRAL = 10  # Cuántos frames seguidos debe ver la acción para activarse
ultima_accion_detectada = "reposo"

print("\n🚀 ¡Sistema en marcha! Párate frente a la cámara.")
print("Usa tu movimiento de 'aplaudir' para iniciar/detener el escaneo.")

while True:
    ret, frame = camara.leer_frame()
    if not ret or frame is None:
        continue

    # Procesar con MediaPipe de forma óptima
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = pose.process(frame_rgb)

    accion_actual = "reposo"

    if resultados.pose_landmarks:
        # Extraer coordenadas actuales
        frame_landmarks = []
        for lm in resultados.pose_landmarks.landmark:
            frame_landmarks.extend([lm.x, lm.y, lm.z])
        
        # Guardar en nuestro buffer temporal
        buffer_frames.append(frame_landmarks)

        # Si ya completamos los 30 frames requeridos, la IA empieza a predecir
        if len(buffer_frames) == LARGO_SECUENCIA:
            entrada_ia = np.expand_dims(buffer_frames, axis=0) # Ajustar dimensiones para Keras (1, 30, 99)
            prediccion = modelo.predict(entrada_ia, verbose=0)[0]
            
            # Obtener el índice con mayor probabilidad
            indice_predicho = np.argmax(prediccion)
            confianza = prediccion[indice_predicho]

            # Solo tomamos en cuenta la predicción si la IA está muy segura (> 75%)
            if confianza > 0.75:
                accion_actual = ACCIONES[indice_predicho]

    # --- LÓGICA DEL INTERRUPTOR (GATILLO POR MOVIMIENTO) ---
    if accion_actual == "aplaudir":
        contador_confirmacion += 1
        # Si mantiene el aplauso un momento continuo
        if contador_confirmacion >= CONFIRMACION_UMBRAL:
            escaneo_activo = not escaneo_activo  # <-- Corrección limpia sin referencias a 'self'
            contador_confirmacion = 0
            print(f"🔄 ¡INTERRUPTOR ACTIVADO! Estado del escaneo: {escaneo_activo}")
            time.sleep(1) # Pequeña pausa de cortesía para que el usuario baje las manos
    else:
        if contador_confirmacion > 0:
            contador_confirmacion -= 1

    # --- INTERFAZ GRÁFICA EN VIVO ---
    frame_visual = cv2.flip(frame, 1) # Efecto espejo en pantalla
    
    # Color de la interfaz según el estado del escaneo
    color_interfaz = (0, 255, 0) if escaneo_activo else (0, 0, 255)
    texto_estado = "ESCANEO: ACTIVO" if escaneo_activo else "ESCANEO: DETENIDO"
    
    # Dibujar info en pantalla
    cv2.putText(frame_visual, f"IA detecto: {accion_actual.upper()}", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame_visual, texto_estado, (15, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_interfaz, 2)
    
    if escaneo_activo:
        cv2.putText(frame_visual, " Analizando flujo de datos...", (15, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow('Detector de Movimiento por IA', frame_visual)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpieza limpia de recursos
camara.liberar()
cv2.destroyAllWindows()
pose.close()