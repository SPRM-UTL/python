"""
Detector de movimiento personalizado.

Cada vez que grabas el mismo movimiento ([r] + mismo nombre), se añade una
muestra y la detección mejora (compara contra todas + un prototipo promedio).

Teclas:
  [r]  Grabar / añadir muestra (nombre en pantalla, Enter confirma)
  [d]  Activar / pausar detección
  [q]  Salir

En el dialogo de nombre: Tab alterna movimientos existentes, Esc cancela.
"""
import sys
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from movimiento.ui_camara import cuenta_regresiva, pedir_nombre_movimiento
from movimiento.utilidades_pose import (
    cargar_referencias,
    guardar_referencia,
    landmarks_a_vector,
    mejor_similitud,
    resumen_referencias,
    umbral_deteccion,
)

LARGO_SECUENCIA = 30
COOLDOWN_SEGUNDOS = 2.5
CUENTA_REGRESIVA_SEG = 3


def _beep():
    try:
        import winsound

        winsound.Beep(880, 200)
    except Exception:
        print("\a", end="", flush=True)


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


def grabar_secuencia(cap, pose) -> list | None:
    secuencia = []
    frames_sin_cuerpo = 0

    while len(secuencia) < LARGO_SECUENCIA:
        ret, frame = cap.read()
        if not ret:
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = pose.process(frame_rgb)
        vista = cv2.flip(frame, 1)

        if resultados.pose_landmarks:
            secuencia.append(landmarks_a_vector(resultados.pose_landmarks))
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
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    referencias = cargar_referencias()
    buffer = deque(maxlen=LARGO_SECUENCIA)
    deteccion_activa = bool(referencias)
    ultimo_aviso = 0.0
    ultimo_match = ""
    estado_ui = "LISTO"

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
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = pose.process(frame_rgb)
        vista = cv2.flip(frame, 1)

        if resultados.pose_landmarks:
            buffer.append(landmarks_a_vector(resultados.pose_landmarks))

            if deteccion_activa and referencias and len(buffer) == LARGO_SECUENCIA:
                actual = np.array(buffer, dtype=np.float32)
                mejor_nombre = ""
                mejor_score = 0.0
                umbral_actual = 0.0

                for nombre, muestras in referencias.items():
                    score = mejor_similitud(muestras, actual)
                    if score > mejor_score:
                        mejor_score = score
                        mejor_nombre = nombre
                        umbral_actual = umbral_deteccion(len(muestras))

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

        lineas = [
            (f"Estado: {estado_ui}", (255, 255, 255)),
            (
                f"Detección: {'ACTIVA' if deteccion_activa else 'PAUSADA'}",
                (0, 255, 0) if deteccion_activa else (0, 0, 255),
            ),
        ]

        if referencias:
            for nombre, muestras in referencias.items():
                lineas.append((f"  {nombre}: {len(muestras)} muestra(s)", (180, 180, 180)))
        else:
            lineas.append(("Sin referencias - pulsa [r]", (0, 165, 255)))

        if time.time() - ultimo_aviso < 1.5 and ultimo_match:
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
            nombre = pedir_nombre_movimiento(cap, list(referencias.keys()))
            if not nombre:
                print("Grabacion cancelada.")
                continue

            nombre_key = "".join(
                c if c.isalnum() or c in "-_" else "_" for c in nombre.strip()
            ).lower() or "movimiento"
            n_antes = len(referencias.get(nombre_key, []))
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
            secuencia = grabar_secuencia(cap, pose)
            if secuencia:
                ruta, total = guardar_referencia(nombre, secuencia)
                referencias = cargar_referencias()
                deteccion_activa = True
                buffer.clear()
                estado_ui = f"GUARDADO: {nombre_key} ({total})"
                print(f"Guardado: {ruta}")
                print(f"  Total muestras de '{nombre_key}': {total}")
                if total < 3:
                    print("  Consejo: graba 2-3 veces más el mismo movimiento con [r].")
                else:
                    print("  Prueba repetir el movimiento frente a la cámara.")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    main()
