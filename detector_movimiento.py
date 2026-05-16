"""
Detector de movimiento personalizado.

Cada vez que grabas el mismo movimiento ([r] + mismo nombre), se añade una
muestra y la detección mejora (compara contra todas + un prototipo promedio).

Teclas:
  [r]  Grabar / añadir muestra (nombre en pantalla, Enter confirma)
  [d]  Activar / pausar detección
  [q]  Salir

En el dialogo de grabacion: selector de movimientos existentes o nuevo, Esc cancela.
"""
import sys
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from movimiento.ui_camara import cuenta_regresiva, pedir_nombre_movimiento
from movimiento.utilidades_pose import (
    buscar_clave_por_nombre,
    cargar_referencias,
    detectar_mejor,
    frame_a_vector_completo,
    guardar_referencia,
    nombre_visible,
    preparar_indice_deteccion,
    resumen_extras,
    resumen_referencias,
)

LARGO_SECUENCIA = 30
COOLDOWN_SEGUNDOS = 2.5
CUENTA_REGRESIVA_SEG = 3
INTERVALO_DETECCION = 3


def _beep():
    try:
        import winsound

        winsound.Beep(880, 200)
    except Exception:
        print("\a", end="", flush=True)


def _formatear_extras_ui(extras: list[float]) -> list[tuple[str, tuple]]:
    info = resumen_extras(extras)
    dedos_izq = int(info["dedos_izq"]) if info["dedos_izq"] >= 0 else "?"
    dedos_der = int(info["dedos_der"]) if info["dedos_der"] >= 0 else "?"
    return [
        (f"Dedos: Izq {dedos_izq}  Der {dedos_der}", (200, 255, 200)),
        (
            f"Inclinacion: cuerpo {info['incl_cuerpo']:+.0f}  hombros {info['incl_hombros']:+.0f}",
            (200, 220, 255),
        ),
        (
            f"              cabeza {info['incl_cabeza']:+.0f}  brazo {info['incl_brazo']:+.0f}",
            (200, 220, 255),
        ),
    ]


def _dibujar_texto(frame, lineas, y_inicio=30):
    for i, (texto, color) in enumerate(lineas):
        cv2.putText(
            frame,
            texto,
            (15, y_inicio + i * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )


def grabar_secuencia(cap, pose, hands) -> list | None:
    secuencia = []
    frames_sin_cuerpo = 0

    while len(secuencia) < LARGO_SECUENCIA:
        ret, frame = cap.read()
        if not ret:
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = pose.process(frame_rgb)
        resultados_manos = hands.process(frame_rgb)
        vista = cv2.flip(frame, 1)

        if resultados.pose_landmarks:
            vector = frame_a_vector_completo(
                resultados.pose_landmarks, resultados_manos
            )
            secuencia.append(vector)
            frames_sin_cuerpo = 0
            progreso = f"GRABANDO {len(secuencia)}/{LARGO_SECUENCIA}"
            color = (0, 0, 255)
        else:
            frames_sin_cuerpo += 1
            progreso = "COLOCATE FRENTE A LA CAMARA"
            color = (0, 165, 255)
            if frames_sin_cuerpo > 90:
                print("Grabación cancelada: no se detectó cuerpo.")
                return None

        _dibujar_texto(vista, [(progreso, color)])
        cv2.imshow("Detector de movimiento", vista)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None

    return secuencia


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    referencias = cargar_referencias()
    indice_deteccion = preparar_indice_deteccion(referencias)
    buffer = deque(maxlen=LARGO_SECUENCIA)
    deteccion_activa = bool(referencias)
    ultimo_aviso = 0.0
    ultimo_match = ""
    estado_ui = "LISTO"
    contador_frames = 0

    print("=" * 55)
    print("  DETECTOR DE MOVIMIENTO PERSONALIZADO")
    print("=" * 55)
    print(f"Movimientos: {resumen_referencias(referencias)}")
    print()
    print("  [r] Grabar movimiento (repite el nombre para añadir muestras)")
    print("  [d] Activar / pausar detección")
    print("  [q] Salir")
    print()
    print("  Tip: graba el mismo movimiento 3-5 veces para mejorar la detección.")
    print("  Incluye dedos e inclinacion: re-graba gestos antiguos para mayor precision.")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = pose.process(frame_rgb)
        resultados_manos = hands.process(frame_rgb)
        vista = cv2.flip(frame, 1)
        extras_actuales = None

        if resultados.pose_landmarks:
            vector = frame_a_vector_completo(
                resultados.pose_landmarks, resultados_manos
            )
            extras_actuales = vector[-6:]
            buffer.append(vector)

            if (
                deteccion_activa
                and indice_deteccion
                and len(buffer) == LARGO_SECUENCIA
            ):
                contador_frames += 1
                if contador_frames % INTERVALO_DETECCION == 0:
                    actual = np.array(buffer, dtype=np.float32)
                    mejor_nombre, mejor_score, umbral_actual = detectar_mejor(
                        indice_deteccion, actual
                    )
                    ahora = time.time()
                    if (
                        mejor_nombre
                        and mejor_score >= umbral_actual
                        and ahora - ultimo_aviso > COOLDOWN_SEGUNDOS
                    ):
                        ultimo_aviso = ahora
                        ultimo_match = mejor_nombre
                        estado_ui = f"DETECTADO: {mejor_nombre.upper()}"
                        print(f"\n[{mejor_nombre}] detectado")
                        _beep()
        else:
            buffer.clear()

        reciente = time.time() - ultimo_aviso < 1.5

        lineas = [
            (f"Estado: {estado_ui}", (255, 255, 255)),
            (
                f"Detección: {'ACTIVA' if deteccion_activa else 'PAUSADA'}",
                (0, 255, 0) if deteccion_activa else (0, 0, 255),
            ),
        ]

        if ultimo_match:
            color_ultimo = (0, 255, 0) if reciente else (0, 220, 200)
            lineas.append((f"Último detectado: {ultimo_match}", color_ultimo))

        if extras_actuales is not None:
            lineas.extend(_formatear_extras_ui(extras_actuales))

        if reciente and ultimo_match:
            cv2.rectangle(vista, (0, 0), (vista.shape[1], vista.shape[0]), (0, 255, 0), 12)
            lineas.insert(0, (f"{ultimo_match.upper()} DETECTADO", (0, 255, 0)))

        _dibujar_texto(vista, lineas)
        cv2.imshow("Detector de movimiento", vista)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break

        if tecla == ord("d"):
            if not referencias:
                print("Primero graba al menos un movimiento con [r].")
            else:
                deteccion_activa = not deteccion_activa
                estado_ui = "DETECTANDO..." if deteccion_activa else "PAUSADO"
                print(f"Detección: {'ACTIVA' if deteccion_activa else 'PAUSADA'}")

        if tecla == ord("r"):
            movimientos_ui = [
                (nombre_visible(k), len(referencias[k]))
                for k in sorted(referencias, key=nombre_visible)
            ]
            nombre = pedir_nombre_movimiento(cap, movimientos_ui)
            if not nombre:
                print("Grabacion cancelada.")
                continue

            clave = buscar_clave_por_nombre(referencias, nombre)
            nombre_key = nombre_visible(clave) if clave else nombre.strip().lower()
            n_antes = len(referencias.get(clave, [])) if clave else 0
            numero_muestra = n_antes + 1

            if not cuenta_regresiva(
                cap,
                nombre,
                segundos=CUENTA_REGRESIVA_SEG,
                numero_muestra=numero_muestra,
            ):
                print("Grabacion cancelada.")
                continue

            print("Grabando... manten el movimiento ~1 segundo.")
            secuencia = grabar_secuencia(cap, pose, hands)
            if secuencia:
                ruta, total, clave = guardar_referencia(nombre, secuencia, clave_existente=clave)
                referencias = cargar_referencias()
                indice_deteccion = preparar_indice_deteccion(referencias)
                deteccion_activa = True
                buffer.clear()
                contador_frames = 0
                estado_ui = f"GUARDADO: {nombre_key} ({total})"
                print(f"Guardado: {ruta}")
                print(f"  Clave: {clave}")
                print(f"  Total muestras de '{nombre_key}': {total}")
                if total < 3:
                    print("  Consejo: graba 2-3 veces más el mismo movimiento con [r].")
                else:
                    print("  Prueba repetir el movimiento frente a la cámara.")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    hands.close()


if __name__ == "__main__":
    main()
