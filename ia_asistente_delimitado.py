"""
Asistente de captura y entrenamiento de gestos.
Pipeline unificado: MediaPipe normalizado + LSTM + similitud + UI en pantalla.
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
from movimiento.config import CUENTA_REGRESIVA_SEG, LARGO_SECUENCIA, MIN_FRAMES_CAPTURA
from movimiento.deteccion import predecir_gesto
from movimiento.ia_lstm import cargar_o_entrenar, entrenar_desde_referencias
from movimiento.ui_camara import (
    VENTANA,
    aviso_breve,
    cuenta_regresiva,
    hud_grabando,
    hud_listo,
    hud_procesando,
    pedir_accion_post_captura,
)
from movimiento.utilidades_pose import (
    buscar_clave_por_nombre,
    cargar_referencias,
    guardar_referencia,
    nombre_visible,
    preparar_secuencia_captura,
)


def _total_muestras(referencias: dict) -> int:
    return sum(len(m) for m in referencias.values())


def _resumen_corto(referencias: dict) -> str:
    if not referencias:
        return "(ninguno — crea el primero con [G] tras capturar)"
    partes = [f"{nombre_visible(k)}:{len(m)}" for k, m in referencias.items()]
    return ", ".join(partes)


def main():
    referencias = cargar_referencias()
    modelo = cargar_o_entrenar()
    sesion = SesionCaptura()
    cap = sesion.abrir_camara()

    buffer: list = []
    grabando = False

    print("\n" + "=" * 70)
    print("  ASISTENTE DE GESTOS — captura con normalizacion y veredicto hibrido")
    print("=" * 70)
    print("  [Espacio] Iniciar (cuenta atras) / Detener y analizar")
    print("  Tras captura: menu en pantalla [S] confirmar [N] corregir [G] nuevo [C] descartar")
    print("  [Q] Salir")
    print("-" * 70 + "\n")

    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        vista, res_pose, res_manos = sesion.procesar_frame(frame)
        sesion.dibujar_esqueleto(vista, res_pose, res_manos)

        referencias = cargar_referencias()
        total = _total_muestras(referencias)
        resumen = _resumen_corto(referencias)

        if grabando:
            vector = sesion.vector_normalizado(res_pose, res_manos)
            pose_ok = vector is not None
            if pose_ok:
                buffer.append(vector)
            hud_grabando(vista, len(buffer), pose_ok)
        else:
            hud_listo(vista, resumen, total)

        cv2.imshow(VENTANA, vista)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q"):
            break

        if tecla != ord(" "):
            continue

        if not grabando:
            sesion.reiniciar_suavizado()
            if not cuenta_regresiva(cap, "movimiento", segundos=CUENTA_REGRESIVA_SEG):
                continue
            buffer.clear()
            grabando = True
            continue

        grabando = False

        if len(buffer) < MIN_FRAMES_CAPTURA:
            aviso_breve(
                cap,
                f"Muy corto ({len(buffer)} frames). Minimo {MIN_FRAMES_CAPTURA}.",
            )
            buffer.clear()
            sesion.reiniciar_suavizado()
            continue

        hud_procesando(vista, "Normalizando secuencia...")
        try:
            secuencia = preparar_secuencia_captura(buffer, LARGO_SECUENCIA)
        except ValueError:
            aviso_breve(cap, "Error: sin datos validos en la captura.")
            buffer.clear()
            continue

        frames_origen = len(buffer)
        buffer.clear()
        sesion.reiniciar_suavizado()

        referencias = cargar_referencias()
        resultado = predecir_gesto(modelo, referencias, secuencia)

        print(
            f"\n[IA] {resultado.nombre} ({resultado.confianza * 100:.1f}%) "
            f"origen={resultado.origen} confiable={resultado.confiable} "
            f"margen={resultado.margen:.2f}"
        )

        nombre_final = pedir_accion_post_captura(
            cap, resultado, referencias, frames_origen
        )

        if not nombre_final:
            print("Captura descartada.")
            continue

        referencias = cargar_referencias()
        clave = buscar_clave_por_nombre(referencias, nombre_final)
        ruta, total_muestra, _ = guardar_referencia(
            nombre=nombre_final,
            secuencia=secuencia.tolist(),
            clave_existente=clave,
        )
        print(f"Guardado: {ruta.name} ({total_muestra} muestras de '{nombre_final}')")

        hud_procesando(vista, "Re-entrenando LSTM...")
        modelo = entrenar_desde_referencias(modelo)
        aviso_breve(cap, f"Guardado: {nombre_final}", 1.2)

    cap.release()
    cv2.destroyAllWindows()
    sesion.cerrar()
    print("Sesion finalizada.")


if __name__ == "__main__":
    main()
