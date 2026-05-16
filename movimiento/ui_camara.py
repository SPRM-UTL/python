"""Diálogos y cuenta regresiva dibujados sobre el video de la cámara."""
import math
import time

import cv2
import numpy as np

VENTANA = "Detector de movimiento"


def _beep_corto():
    try:
        import winsound

        winsound.Beep(660, 80)
    except Exception:
        pass


def _beep_inicio():
    try:
        import winsound

        winsound.Beep(990, 120)
    except Exception:
        pass


def _leer_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.flip(frame, 1)


def _oscurecer(vista: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    overlay = vista.copy()
    cv2.rectangle(overlay, (0, 0), (vista.shape[1], vista.shape[0]), (0, 0, 0), -1)
    return cv2.addWeighted(overlay, alpha, vista, 1 - alpha, 0)


def pedir_nombre_movimiento(
    cap,
    nombres_existentes: list[str] | None = None,
) -> str | None:
    """
    Diálogo sobre la cámara para escribir el nombre del movimiento.
    Enter: confirmar | Esc: cancelar | Retroceso: borrar | Tab: nombre existente
    """
    nombre = ""
    indice_sugerencia = -1
    cursor_visible = True
    ultimo_parpadeo = time.time()
    nombres = sorted(nombres_existentes or [])

    while True:
        vista = _leer_frame(cap)
        if vista is None:
            return None

        ahora = time.time()
        if ahora - ultimo_parpadeo > 0.45:
            cursor_visible = not cursor_visible
            ultimo_parpadeo = ahora

        h, w = vista.shape[:2]
        vista = _oscurecer(vista, 0.5)

        panel_w, panel_h = min(520, w - 40), 220
        px, py = (w - panel_w) // 2, (h - panel_h) // 2
        cv2.rectangle(vista, (px, py), (px + panel_w, py + panel_h), (45, 45, 45), -1)
        cv2.rectangle(vista, (px, py), (px + panel_w, py + panel_h), (0, 200, 255), 2)

        cv2.putText(
            vista,
            "NUEVO MOVIMIENTO",
            (px + 20, py + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 220, 255),
            2,
        )
        cv2.putText(
            vista,
            "Escribe el nombre y pulsa Enter",
            (px + 20, py + 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 200, 200),
            1,
        )

        texto = nombre if nombre else "ej: saludar"
        color_texto = (255, 255, 255) if nombre else (120, 120, 120)
        cv2.rectangle(vista, (px + 16, py + 88), (px + panel_w - 16, py + 130), (30, 30, 30), -1)
        cv2.rectangle(vista, (px + 16, py + 88), (px + panel_w - 16, py + 130), (80, 80, 80), 1)

        mostrar = texto
        if nombre and cursor_visible:
            mostrar = nombre + "|"
        cv2.putText(
            vista,
            mostrar[:28],
            (px + 28, py + 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color_texto,
            2,
        )

        ayuda = "Enter guardar   Esc cancelar   Tab nombre previo"
        cv2.putText(
            vista,
            ayuda,
            (px + 20, py + 158),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (160, 160, 160),
            1,
        )

        if nombres:
            lista = "Existentes: " + ", ".join(nombres[:5])
            if len(nombres) > 5:
                lista += "..."
            cv2.putText(
                vista,
                lista,
                (px + 20, py + 188),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (100, 200, 100),
                1,
            )

        cv2.imshow(VENTANA, vista)
        tecla = cv2.waitKey(30) & 0xFF

        if tecla in (27, ord("q")):
            return None
        if tecla in (13, 10):
            limpio = nombre.strip()
            return limpio if limpio else None
        if tecla in (8, 127):
            nombre = nombre[:-1]
            indice_sugerencia = -1
        elif tecla == 9 and nombres:
            indice_sugerencia = (indice_sugerencia + 1) % len(nombres)
            nombre = nombres[indice_sugerencia]
        elif 32 <= tecla <= 126 and len(nombre) < 28:
            nombre += chr(tecla)


def cuenta_regresiva(
    cap,
    nombre: str,
    segundos: int = 3,
    numero_muestra: int | None = None,
) -> bool:
    """
    Cuenta regresiva animada sobre la cámara.
    Devuelve False si el usuario cancela con Esc o q.
    """
    inicio = time.time()
    duracion = float(segundos)
    ultimo_entero = segundos + 1

    while True:
        transcurrido = time.time() - inicio
        restante = duracion - transcurrido

        if restante <= 0:
            break

        entero = int(math.ceil(restante))
        if entero != ultimo_entero and entero <= segundos:
            ultimo_entero = entero
            _beep_corto()

        vista = _leer_frame(cap)
        if vista is None:
            return False

        h, w = vista.shape[:2]
        vista = _oscurecer(vista, 0.35)

        cx, cy = w // 2, h // 2
        progreso = 1.0 - (restante / duracion)
        radio = 100

        cv2.ellipse(
            vista,
            (cx, cy),
            (radio, radio),
            -90,
            0,
            360,
            (60, 60, 60),
            6,
        )
        angulo_fin = int(360 * progreso)
        if angulo_fin > 0:
            cv2.ellipse(
                vista,
                (cx, cy),
                (radio, radio),
                -90,
                0,
                angulo_fin,
                (0, 220, 255),
                6,
            )

        escala_num = 3.2 + 0.25 * math.sin(transcurrido * 8)
        texto_num = str(entero)
        tam = cv2.getTextSize(texto_num, cv2.FONT_HERSHEY_DUPLEX, escala_num, 4)[0]
        cv2.putText(
            vista,
            texto_num,
            (cx - tam[0] // 2, cy + tam[1] // 2),
            cv2.FONT_HERSHEY_DUPLEX,
            escala_num,
            (0, 80, 255),
            4,
        )

        titulo = f"Preparate: {nombre}"
        cv2.putText(
            vista,
            titulo[:40],
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if numero_muestra is not None:
            cv2.putText(
                vista,
                f"Muestra #{numero_muestra}",
                (20, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 200),
                2,
            )

        barra_w = int((w - 80) * progreso)
        cv2.rectangle(vista, (40, h - 36), (w - 40, h - 20), (50, 50, 50), -1)
        if barra_w > 0:
            cv2.rectangle(vista, (40, h - 36), (40 + barra_w, h - 20), (0, 200, 255), -1)

        cv2.putText(
            vista,
            "Esc cancelar",
            (20, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 150, 150),
            1,
        )

        cv2.imshow(VENTANA, vista)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (27, ord("q")):
            return False

    _beep_inicio()
    fin_go = time.time() + 0.45
    while time.time() < fin_go:
        vista = _leer_frame(cap)
        if vista is None:
            return False
        h, w = vista.shape[:2]
        vista = _oscurecer(vista, 0.25)
        tam = cv2.getTextSize("YA!", cv2.FONT_HERSHEY_DUPLEX, 2.8, 5)[0]
        cv2.putText(
            vista,
            "YA!",
            (w // 2 - tam[0] // 2, h // 2 + tam[1] // 2),
            cv2.FONT_HERSHEY_DUPLEX,
            2.8,
            (0, 255, 100),
            5,
        )
        cv2.imshow(VENTANA, vista)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            return False

    return True
