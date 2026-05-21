"""
Asistente por videos: grabar, almacenar por gesto, analizar y entrenar la IA.

Uso:
  python asistente_videos.py

Flujo recomendado:
  1. Grabar varios videos por gesto (distancias/velocidades distintas)
  2. Analizar videos -> extrae secuencias a referencias/
  3. Entrenar IA
"""
import os
import sys
import types

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2

try:
    import google.protobuf.runtime_version  # noqa: F401
except ImportError:
    mod = types.ModuleType("google.protobuf.runtime_version")
    mod.runtime_version = lambda *args, **kwargs: None
    mod.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None
    sys.modules["google.protobuf.runtime_version"] = mod

from movimiento.captura_pose import SesionCaptura
from movimiento.config import VENTANA_VIDEOS
from movimiento.ia_lstm import cargar_o_entrenar, entrenar_desde_referencias
from movimiento.ui_camara import aviso_breve, hud_procesando
from movimiento.ui_videos import (
    flujo_analizar,
    flujo_grabar_videos,
    mostrar_biblioteca,
    pedir_opcion_menu,
)


def _entrenar(cap, modelo):
    vista = _leer_frame_local(cap)
    if vista is not None:
        hud_procesando(vista, "Entrenando LSTM...")
        cv2.imshow(VENTANA_VIDEOS, vista)
    print("\n[IA] Entrenando...")
    modelo = entrenar_desde_referencias(modelo)
    aviso_breve(cap, "Modelo actualizado", 1.5)
    return modelo


def _leer_frame_local(cap):
    from movimiento.ui_camara import _leer_frame
    return _leer_frame(cap)


def main():
    import movimiento.ui_camara as ui_camara

    ui_camara.VENTANA = VENTANA_VIDEOS

    modelo = cargar_o_entrenar()
    sesion = SesionCaptura()
    cap = sesion.abrir_camara()

    cv2.namedWindow(VENTANA_VIDEOS, cv2.WINDOW_NORMAL)

    print("\n" + "=" * 70)
    print("  ASISTENTE DE VIDEOS — grabar, analizar y mejorar la IA")
    print("=" * 70)
    print("  Los videos se guardan en: videos/<gesto>/video_XXX.mp4")
    print("  El analisis genera muestras en: referencias/<gesto>/")
    print("-" * 70 + "\n")

    while cap.isOpened():
        opcion = pedir_opcion_menu(cap)
        if opcion is None or opcion == "q":
            break

        if opcion == "1":
            flujo_grabar_videos(cap, sesion)

        elif opcion == "2":
            mostrar_biblioteca(cap)

        elif opcion == "3":
            if flujo_analizar(cap, sesion):
                modelo = _entrenar(cap, modelo)

        elif opcion == "4":
            modelo = _entrenar(cap, modelo)

    cap.release()
    cv2.destroyAllWindows()
    sesion.cerrar()
    print("Sesion finalizada.")


if __name__ == "__main__":
    main()
