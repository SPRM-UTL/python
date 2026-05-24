"""MediaPipe: captura de vectores normalizados (espejo + suavizado)."""
import sys

import cv2
import mediapipe as mp
import numpy as np

from movimiento.config import (
    CAMARA_ALTO,
    CAMARA_ANCHO,
    CAMARA_BUFFER,
    CAMARA_FOURCC,
    CAMARA_FPS,
    CAMARA_INDICE,
    CAMARA_RESOLUCIONES,
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
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
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

    def dibujar_esqueleto(self, vista, resultados_pose, resultados_manos) -> None:
        if resultados_pose.pose_landmarks:
            self._mp_drawing.draw_landmarks(
                vista,
                resultados_pose.pose_landmarks,
                self._mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self._mp_styles.get_default_pose_landmarks_style(),
            )
        if resultados_manos and resultados_manos.multi_hand_landmarks:
            for hand_lm in resultados_manos.multi_hand_landmarks:
                self._mp_drawing.draw_landmarks(
                    vista,
                    hand_lm,
                    self._mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self._mp_styles.get_default_hand_landmarks_style(),
                )

    @staticmethod
    def _backend_captura() -> int | None:
        if sys.platform == "win32" and hasattr(cv2, "CAP_DSHOW"):
            return cv2.CAP_DSHOW
        if sys.platform == "win32" and hasattr(cv2, "CAP_MSMF"):
            return cv2.CAP_MSMF
        return None

    @staticmethod
    def _aplicar_fourcc(cap: cv2.VideoCapture, fourcc: str) -> None:
        if len(fourcc) == 4:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

    @staticmethod
    def configurar_camara(cap: cv2.VideoCapture) -> tuple[int, int, float]:
        """
        Ajusta resolución, FPS y buffer. Devuelve (ancho, alto, fps) reales.
        """
        SesionCaptura._aplicar_fourcc(cap, CAMARA_FOURCC)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMARA_BUFFER)
        if hasattr(cv2, "CAP_PROP_AUTOFOCUS"):
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        ancho, alto = CAMARA_ANCHO, CAMARA_ALTO
        for w, h in CAMARA_RESOLUCIONES:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, CAMARA_FPS)
            # Leer un frame fuerza al driver a aplicar el modo
            cap.grab()
            rw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            rh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if rw >= w * 0.85 and rh >= h * 0.85:
                ancho, alto = rw, rh
                break

        fps = float(cap.get(cv2.CAP_PROP_FPS) or CAMARA_FPS)
        return ancho, alto, fps

    @staticmethod
    def abrir_camara(indice: int = CAMARA_INDICE) -> cv2.VideoCapture:
        backend = SesionCaptura._backend_captura()
        if backend is not None:
            cap = cv2.VideoCapture(indice, backend)
        else:
            cap = cv2.VideoCapture(indice)

        if not cap.isOpened():
            cap = cv2.VideoCapture(indice)

        if not cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la camara (indice {indice}). "
                "Prueba otro CAMARA_INDICE en movimiento/config.py"
            )

        ancho, alto, fps = SesionCaptura.configurar_camara(cap)
        print(f"[Camara] {ancho}x{alto} @ {fps:.0f} FPS ({CAMARA_FOURCC})")
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
