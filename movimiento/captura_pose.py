"""MediaPipe: captura de vectores normalizados (espejo + suavizado)."""
import cv2
import mediapipe as mp
import numpy as np

from movimiento.config import (
    CAMARA_ALTO,
    CAMARA_ANCHO,
    HANDS_COMPLEXITY,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    POSE_COMPLEXITY,
    SUAVIZADO_ALPHA,
)
from movimiento.utilidades_pose import frame_a_vector_completo, normalizar_frame


class SesionCaptura:
    def __init__(self):
        self._mp_pose = mp.solutions.pose
        self._mp_hands = mp.solutions.hands
        self.pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=POSE_COMPLEXITY,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self.hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=HANDS_COMPLEXITY,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self._vector_anterior: np.ndarray | None = None

    @staticmethod
    def abrir_camara(indice: int = 0) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(indice)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMARA_ANCHO)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMARA_ALTO)
        return cap

    def reiniciar_suavizado(self) -> None:
        self._vector_anterior = None

    def procesar_frame(self, frame_bgr: np.ndarray):
        """Devuelve (vista_espejo, resultados_pose, resultados_manos)."""
        vista = cv2.flip(frame_bgr, 1)
        rgb = cv2.cvtColor(vista, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        resultados_pose = self.pose.process(rgb)
        resultados_manos = self.hands.process(rgb)
        return vista, resultados_pose, resultados_manos

    def vector_normalizado(self, resultados_pose, resultados_manos) -> np.ndarray | None:
        if not resultados_pose.pose_landmarks:
            return None
        crudo = np.array(
            frame_a_vector_completo(resultados_pose.pose_landmarks, resultados_manos),
            dtype=np.float32,
        )
        vector = normalizar_frame(crudo)
        if self._vector_anterior is not None:
            vector = (
                SUAVIZADO_ALPHA * vector + (1.0 - SUAVIZADO_ALPHA) * self._vector_anterior
            ).astype(np.float32)
        self._vector_anterior = vector.copy()
        return vector

    def cerrar(self) -> None:
        self.pose.close()
        self.hands.close()
