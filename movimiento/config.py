"""Parámetros del asistente de gestos (único pipeline activo)."""

LARGO_SECUENCIA = 30
MODEL_NAME = "ia_detector_referencias.h5"

MIN_FRAMES_CAPTURA = 8
FRAMES_IDEAL_MIN = 15
FRAMES_IDEAL_MAX = 50
EPOCHS_ENTRENAMIENTO = 15
BATCH_SIZE = 2
UMBRAL_CONFIANZA_LSTM = 0.68
MARGEN_MINIMO_DETECCION = 0.10
REQUIERE_ACUERDO_LSTM_SIM = True
CUENTA_REGRESIVA_SEG = 3
MUESTRAS_OBJETIVO_POR_GESTO = 8

# MediaPipe: 2 = más precisión (recomendado con HD); 1 si va lento en tu PC
POSE_COMPLEXITY = 2
HANDS_COMPLEXITY = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# Cámara: se intentan en orden hasta que el dispositivo las acepte
CAMARA_RESOLUCIONES = (
    (1280, 720),
    (1920, 1080),
    (960, 540),
    (640, 480),
)
CAMARA_ANCHO = 1280
CAMARA_ALTO = 720
CAMARA_FPS = 30
CAMARA_INDICE = 0
CAMARA_FOURCC = "MJPG"  # muchas webcams USB entregan mejor HD con MJPEG
CAMARA_BUFFER = 1  # menos latencia y frames "viejos"
SUAVIZADO_ALPHA = 0.35

VENTANA_ASISTENTE = "Asistente de gestos"
VENTANA_VIDEOS = "Biblioteca de videos"

# Videos de entrenamiento (por gesto)
CARPETA_VIDEOS = "videos"
VIDEO_FPS = 30
VIDEO_CODEC = "mp4v"
FRAMES_VIDEO_AUTO = LARGO_SECUENCIA  # la grabacion termina sola al llegar aqui
MIN_FRAMES_VIDEO = LARGO_SECUENCIA
VENTANA_VIDEO_PASO = 10  # stride al extraer secuencias de un video
MIN_FRAMES_VALIDOS_RATIO = 0.8  # % de frames con pose en una ventana
MAX_MUESTRAS_POR_VIDEO = 12
UMBRAL_DEDUPE_MUESTRA = 0.92  # no guardar si ya hay una casi igual
