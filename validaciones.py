import cv2
import mediapipe as mp
import numpy as np

class AnalizadorHumano:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands

        # Optimizamos Pose con model_complexity=0 o 1 (0 es el más rápido)
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, 
            model_complexity=0,       # <-- CLAVE: Menor complejidad = Más FPS
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        
        # Eliminamos Face y Segmentation. Dejamos Hands optimizado.
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=2, 
            model_complexity=0,       # <-- CLAVE: Más rápido
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )

        self.resultados_pose = None
        self.resultados_hands = None

    def actualizar_frame(self, frame):
        """
        Convierte, redimensiona y procesa únicamente lo necesario.
        """
        # Redimensionar el frame para el procesamiento de IA (Mejora brutal de rendimiento)
        # Ajusta 640x480 o similar según tus necesidades de cámara
        h, w = frame.shape[:2]
        if w > 640:
            frame_pequeno = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        else:
            frame_pequeno = frame

        frame_rgb = cv2.cvtColor(frame_pequeno, cv2.COLOR_BGR2RGB)
        
        # Procesamos solo 2 modelos en lugar de 4
        self.resultados_pose = self.pose.process(frame_rgb)
        self.resultados_hands = self.hands.process(frame_rgb)

    def _obtener_landmarks_pose(self):
        if self.resultados_pose and self.resultados_pose.pose_landmarks:
            return self.resultados_pose.pose_landmarks.landmark
        return None

    # --- MÉTODOS REFACTORIZADOS Y OPTIMIZADOS ---

    def detectar_humano(self):
        # Si Pose detecta landmarks con suficiente confianza, hay un humano.
        # No necesitas segmentación por píxeles para esto.
        return self._obtener_landmarks_pose() is not None

    def detectar_cara(self):
        # Pose ya incluye la nariz y los ojos. Si están visibles, la cara está ahí.
        lm = self._obtener_landmarks_pose()
        if not lm: return False
        # Validamos si la nariz tiene buena visibilidad
        return lm[self.mp_pose.PoseLandmark.NOSE].visibility > 0.5

    def esta_de_frente(self):
        lm = self._obtener_landmarks_pose()
        if not lm: return False
        
        # Rostro visible (ojos/orejas) suele indicar que está de frente o de lado, no de espaldas
        oreja_izq = lm[self.mp_pose.PoseLandmark.LEFT_EAR].visibility > 0.5
        oreja_der = lm[self.mp_pose.PoseLandmark.RIGHT_EAR].visibility > 0.5
        
        if not (oreja_izq and oreja_der): return False

        centro_hombros = (lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x + lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2
        return abs(lm[self.mp_pose.PoseLandmark.NOSE].x - centro_hombros) < 0.08

    def esta_de_espalda(self):
        lm = self._obtener_landmarks_pose()
        if not lm: return False
        # Si el cuerpo está pero las orejas/ojos/nariz no son visibles, está de espaldas
        return lm[self.mp_pose.PoseLandmark.NOSE].visibility <= 0.3

    def esta_de_lado_izquierdo(self):
        lm = self._obtener_landmarks_pose()
        if not lm: return False
        centro_hombros = (lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x + lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2
        return lm[self.mp_pose.PoseLandmark.NOSE].x < centro_hombros - 0.08

    def esta_de_lado_derecho(self):
        lm = self._obtener_landmarks_pose()
        if not lm: return False
        centro_hombros = (lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x + lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2
        return lm[self.mp_pose.PoseLandmark.NOSE].x > centro_hombros + 0.08

    def _detectar_mano_por_etiqueta(self, etiqueta_buscada):
        if not self.resultados_hands or not self.resultados_hands.multi_handedness:
            return False
        for mano in self.resultados_hands.multi_handedness:
            if mano.classification[0].label == etiqueta_buscada:
                return True
        return False

    def detectar_mano_izquierda(self):
        return self._detectar_mano_por_etiqueta("Left")

    def detectar_mano_derecha(self):
        return self._detectar_mano_por_etiqueta("Right")

    def _validar_visibilidad_puntos(self, puntos, min_visibles):
        lm = self._obtener_landmarks_pose()
        if not lm: return False
        visibles = sum(1 for p in puntos if lm[p].visibility > 0.5)
        return visibles >= min_visibles

    def detectar_tronco(self):
        puntos = [self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                  self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP]
        return self._validar_visibilidad_puntos(puntos, 3)

    def detectar_brazo_izquierdo(self):
        puntos = [self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST]
        return self._validar_visibilidad_puntos(puntos, 2)

    def detectar_brazo_derecho(self):
        puntos = [self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST]
        return self._validar_visibilidad_puntos(puntos, 2)

    def detectar_pierna_izquierda(self):
        puntos = [self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE]
        return self._validar_visibilidad_puntos(puntos, 2)

    def detectar_pierna_derecha(self):
        puntos = [self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        return self._validar_visibilidad_puntos(puntos, 2)
