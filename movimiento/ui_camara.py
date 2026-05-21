"""Diálogos y cuenta regresiva dibujados sobre el video de la cámara."""
import math
import time

import cv2
import numpy as np

from movimiento.config import (
    FRAMES_IDEAL_MAX,
    FRAMES_IDEAL_MIN,
    LARGO_SECUENCIA,
    MIN_FRAMES_CAPTURA,
    MUESTRAS_OBJETIVO_POR_GESTO,
    VENTANA_ASISTENTE,
)

VENTANA = VENTANA_ASISTENTE


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


def _tecla_flecha(tecla: int) -> str | None:
    if tecla in (65362, 2490368, ord("w"), ord("W"), ord("k"), ord("K")):
        return "arriba"
    if tecla in (65364, 2621440, ord("s"), ord("S"), ord("j"), ord("J")):
        return "abajo"
    return None


def pedir_nombre_movimiento(
    cap,
    movimientos_existentes: list[tuple[str, int]] | None = None,
) -> str | None:
    """
    Selector sobre la cámara: movimiento existente o nombre nuevo.

    Lista: flechas / W-S | Enter: elegir | Esc: cancelar (o volver desde nuevo)
    Nuevo: escribir nombre | Enter: confirmar | Esc: volver a la lista
    """
    existentes = sorted(movimientos_existentes or [], key=lambda x: x[0].lower())
    items: list[tuple[str, int | None]] = [("+ Nuevo movimiento", None)]
    items.extend((nombre, muestras) for nombre, muestras in existentes)

    indice = 0
    scroll = 0
    modo_nuevo = not existentes
    nombre = ""
    cursor_visible = True
    ultimo_parpadeo = time.time()
    max_visibles = 6
    alto_fila = 34

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

        filas_lista = min(len(items), max_visibles)
        panel_w = min(560, w - 40)
        panel_h = 130 + filas_lista * alto_fila + (72 if modo_nuevo else 0)
        px, py = (w - panel_w) // 2, max(20, (h - panel_h) // 2)

        cv2.rectangle(vista, (px, py), (px + panel_w, py + panel_h), (45, 45, 45), -1)
        cv2.rectangle(vista, (px, py), (px + panel_w, py + panel_h), (0, 200, 255), 2)

        cv2.putText(
            vista,
            "GRABAR MOVIMIENTO",
            (px + 20, py + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 220, 255),
            2,
        )
        subtitulo = (
            "Escribe el nombre del movimiento nuevo"
            if modo_nuevo
            else "Elige uno existente o crea uno nuevo"
        )
        cv2.putText(
            vista,
            subtitulo,
            (px + 20, py + 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (180, 180, 180),
            1,
        )

        y_lista = py + 78
        for fila in range(filas_lista):
            idx = scroll + fila
            if idx >= len(items):
                break

            etiqueta, muestras = items[idx]
            y0 = y_lista + fila * alto_fila
            y1 = y0 + alto_fila - 4
            seleccionado = idx == indice and not modo_nuevo

            if seleccionado:
                cv2.rectangle(vista, (px + 14, y0), (px + panel_w - 14, y1), (50, 90, 50), -1)
                cv2.rectangle(vista, (px + 14, y0), (px + panel_w - 14, y1), (0, 220, 120), 2)
                prefijo, color = "> ", (120, 255, 120)
            else:
                prefijo, color = "  ", (200, 200, 200) if modo_nuevo else (170, 170, 170)

            texto = prefijo + etiqueta
            if muestras is not None:
                texto += f"  ({muestras} muestra{'s' if muestras != 1 else ''})"

            cv2.putText(
                vista,
                texto[:42],
                (px + 24, y0 + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2 if seleccionado else 1,
            )

        if len(items) > max_visibles:
            indicador = f"{indice + 1}/{len(items)}"
            cv2.putText(
                vista,
                indicador,
                (px + panel_w - 70, py + panel_h - 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (140, 140, 140),
                1,
            )

        y_campo = y_lista + filas_lista * alto_fila + 8
        if modo_nuevo:
            cv2.putText(
                vista,
                "Nombre nuevo:",
                (px + 20, y_campo),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 220, 255),
                1,
            )
            cv2.rectangle(
                vista,
                (px + 16, y_campo + 10),
                (px + panel_w - 16, y_campo + 48),
                (30, 30, 30),
                -1,
            )
            cv2.rectangle(
                vista,
                (px + 16, y_campo + 10),
                (px + panel_w - 16, y_campo + 48),
                (0, 200, 255),
                1,
            )
            texto = nombre if nombre else "ej: saludar"
            color_texto = (255, 255, 255) if nombre else (120, 120, 120)
            mostrar = texto
            if nombre and cursor_visible:
                mostrar = nombre + "|"
            cv2.putText(
                vista,
                mostrar[:28],
                (px + 28, y_campo + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                color_texto,
                2,
            )

        ayuda = (
            "Enter confirmar   Esc volver"
            if modo_nuevo
            else "Flechas elegir   Enter   Esc cancelar"
        )
        cv2.putText(
            vista,
            ayuda,
            (px + 20, py + panel_h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (140, 140, 140),
            1,
        )

        cv2.imshow(VENTANA, vista)
        tecla = cv2.waitKeyEx(30)
        tecla_ff = tecla & 0xFF

        if tecla_ff in (27, ord("q")):
            if modo_nuevo and existentes:
                modo_nuevo = False
                nombre = ""
                continue
            return None

        if tecla_ff in (13, 10):
            if modo_nuevo:
                limpio = nombre.strip()
                if limpio:
                    return limpio
                continue
            if items[indice][1] is None:
                modo_nuevo = True
                nombre = ""
            else:
                return items[indice][0]
            continue

        if modo_nuevo:
            if tecla_ff in (8, 127):
                nombre = nombre[:-1]
            elif 32 <= tecla_ff <= 126 and len(nombre) < 28:
                nombre += chr(tecla_ff)
            continue

        flecha = _tecla_flecha(tecla if tecla > 255 else tecla_ff)
        if flecha == "arriba":
            indice = (indice - 1) % len(items)
        elif flecha == "abajo":
            indice = (indice + 1) % len(items)
        elif tecla_ff == 9:
            indice = (indice + 1) % len(items)

        if indice < scroll:
            scroll = indice
        elif indice >= scroll + max_visibles:
            scroll = indice - max_visibles + 1


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


def _panel_inferior(vista: np.ndarray, lineas: list[str], color=(220, 220, 220)) -> None:
    h, w = vista.shape[:2]
    alto = 22 + len(lineas) * 22
    y0 = h - alto - 8
    overlay = vista.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, vista, 0.25, 0, vista)
    for i, texto in enumerate(lineas):
        cv2.putText(
            vista,
            texto,
            (14, y0 + 20 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )


def hud_listo(
    vista: np.ndarray,
    resumen_gestos: str,
    total_muestras: int,
) -> None:
    h, w = vista.shape[:2]
    overlay = vista.copy()
    cv2.rectangle(overlay, (0, 0), (w, 56), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.6, vista, 0.4, 0, vista)
    cv2.putText(
        vista,
        "LISTO PARA CAPTURAR",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (80, 255, 120),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vista,
        f"Gestos: {resumen_gestos[:55]}",
        (14, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(vista, (w - 130, 12), (w - 12, 44), (40, 40, 40), -1)
    cv2.putText(
        vista,
        f"{total_muestras} muestras",
        (w - 120, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )
    _panel_inferior(
        vista,
        [
            "[Espacio] Iniciar (cuenta atras)   [Q] Salir",
            f"Objetivo: {MUESTRAS_OBJETIVO_POR_GESTO}+ muestras/gesto, variar distancia y velocidad",
        ],
    )


def hud_grabando(
    vista: np.ndarray,
    num_frames: int,
    pose_ok: bool,
) -> None:
    h, w = vista.shape[:2]
    progreso = min(1.0, num_frames / LARGO_SECUENCIA)
    color_borde = (0, 80, 255) if pose_ok else (0, 80, 255)
    grosor = 8
    cv2.rectangle(vista, (0, 0), (w, h), color_borde, grosor)

    overlay = vista.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (30, 30, 80), -1)
    cv2.addWeighted(overlay, 0.65, vista, 0.35, 0, vista)

    estado = "GRABANDO" if pose_ok else "GRABANDO (sin pose)"
    color_txt = (80, 80, 255) if pose_ok else (80, 160, 255)
    cv2.putText(
        vista,
        estado,
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color_txt,
        2,
        cv2.LINE_AA,
    )

    bar_x, bar_y, bar_w = 14, 40, w - 28
    cv2.rectangle(vista, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (50, 50, 50), -1)
    fill = int(bar_w * progreso)
    if fill > 0:
        cv2.rectangle(vista, (bar_x, bar_y), (bar_x + fill, bar_y + 8), (0, 200, 255), -1)

    calidad = "OK"
    color_cal = (80, 255, 120)
    if num_frames < MIN_FRAMES_CAPTURA:
        calidad = "MUY CORTO"
        color_cal = (80, 80, 255)
    elif num_frames < FRAMES_IDEAL_MIN:
        calidad = "CORTO"
        color_cal = (0, 200, 255)
    elif num_frames > FRAMES_IDEAL_MAX:
        calidad = "LARGO"
        color_cal = (0, 200, 255)

    cv2.putText(
        vista,
        f"{num_frames} frames  |  meta ~{FRAMES_IDEAL_MIN}-{FRAMES_IDEAL_MAX}  |  {calidad}",
        (w - 340, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color_cal,
        1,
        cv2.LINE_AA,
    )
    _panel_inferior(
        vista,
        [
            "[Espacio] Detener y analizar   [Q] Cancelar grabacion",
            f"Se guardaran {LARGO_SECUENCIA} frames normalizados (interpolados si hace falta)",
        ],
    )


def hud_procesando(vista: np.ndarray, mensaje: str = "Procesando...") -> None:
    vista[:] = _oscurecer(vista, 0.45)
    h, w = vista.shape[:2]
    tam = cv2.getTextSize(mensaje, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.putText(
        vista,
        mensaje,
        (w // 2 - tam[0] // 2, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow(VENTANA, vista)
    cv2.waitKey(1)


def pedir_accion_post_captura(
    cap,
    resultado,
    referencias: dict,
    frames_origen: int,
) -> str | None:
    """
    Menú en pantalla tras captura.
    Devuelve nombre del gesto a guardar o None si cancela.
    """
    from movimiento.utilidades_pose import nombre_visible

    nombres = sorted(nombre_visible(k) for k in referencias.keys())
    items_accion = [
        ("Confirmar prediccion", "s"),
        ("Elegir otro gesto", "n"),
        ("Gesto nuevo", "g"),
        ("Descartar", "c"),
    ]
    indice_accion = 0
    indice_gesto = 0
    modo = "accion"
    modo_nuevo = False
    nombre_nuevo = ""

    confiable = resultado.confiable
    puede_confirmar = (
        confiable
        and resultado.nombre
        not in ("Ninguno (Sin modelo)", "Incertidumbre (sin acuerdo)", "Incertidumbre (baja confianza)")
    )

    while True:
        vista = _leer_frame(cap)
        if vista is None:
            return None

        h, w = vista.shape[:2]
        vista = _oscurecer(vista, 0.42)

        panel_w = min(620, w - 30)
        panel_h = 340 if modo == "accion" else 300
        px, py = (w - panel_w) // 2, (h - panel_h) // 2

        cv2.rectangle(vista, (px, py), (px + panel_w, py + panel_h), (40, 40, 40), -1)
        cv2.rectangle(vista, (px, py), (px + panel_w, py + panel_h), (0, 200, 255), 2)

        cv2.putText(
            vista,
            "CAPTURA COMPLETADA",
            (px + 18, py + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 220, 255),
            2,
        )
        cv2.putText(
            vista,
            f"{frames_origen} frames -> {LARGO_SECUENCIA} normalizados",
            (px + 18, py + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 160, 160),
            1,
        )

        color_pred = (120, 255, 120) if confiable else (120, 180, 255)
        cv2.putText(
            vista,
            f"IA: {resultado.nombre}  ({resultado.confianza * 100:.0f}%)",
            (px + 18, py + 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color_pred,
            2,
        )
        cv2.putText(
            vista,
            f"Origen: {resultado.origen}  |  Margen: {resultado.margen:.2f}",
            (px + 18, py + 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 180),
            1,
        )
        if resultado.nombre_lstm:
            cv2.putText(
                vista,
                f"LSTM: {resultado.nombre_lstm} ({resultado.conf_lstm * 100:.0f}%)",
                (px + 18, py + 132),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (140, 140, 200),
                1,
            )
        if resultado.nombre_sim:
            cv2.putText(
                vista,
                f"Sim: {resultado.nombre_sim} ({resultado.score_sim * 100:.0f}%)",
                (px + 18, py + 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (140, 200, 140),
                1,
            )
        if not confiable:
            cv2.putText(
                vista,
                "Baja confianza: confirma solo si estas seguro",
                (px + 18, py + 172),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (100, 180, 255),
                1,
            )

        y0 = py + 195
        if modo == "accion":
            for i, (etiqueta, tecla) in enumerate(items_accion):
                yy = y0 + i * 32
                sel = i == indice_accion
                if etiqueta.startswith("Confirmar") and not puede_confirmar:
                    color = (90, 90, 90)
                    pref = "x "
                else:
                    color = (120, 255, 120) if sel else (200, 200, 200)
                    pref = "> " if sel else "  "
                texto = f"{pref}[{tecla.upper()}] {etiqueta}"
                if etiqueta.startswith("Confirmar") and puede_confirmar:
                    texto += f" -> {resultado.nombre}"
                cv2.putText(
                    vista,
                    texto[:48],
                    (px + 20, yy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    2 if sel else 1,
                )
            ayuda = "Flechas / Enter   Esc cancelar"
        elif modo == "gesto":
            cv2.putText(
                vista,
                "Elige el gesto correcto:",
                (px + 18, y0 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 220, 255),
                1,
            )
            for i, nom in enumerate(nombres[:8]):
                yy = y0 + i * 30
                sel = i == indice_gesto
                color = (120, 255, 120) if sel else (190, 190, 190)
                pref = "> " if sel else "  "
                cv2.putText(
                    vista,
                    f"{pref}{nom}",
                    (px + 24, yy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2 if sel else 1,
                )
            ayuda = "Enter elegir   Esc volver"
        else:
            cv2.putText(
                vista,
                "Nombre del gesto nuevo:",
                (px + 18, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 220, 255),
                1,
            )
            cv2.rectangle(
                vista,
                (px + 16, y0 + 12),
                (px + panel_w - 16, y0 + 48),
                (25, 25, 25),
                -1,
            )
            texto = nombre_nuevo if nombre_nuevo else "ej: reposo"
            cv2.putText(
                vista,
                texto[:32],
                (px + 28, y0 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255) if nombre_nuevo else (120, 120, 120),
                2,
            )
            ayuda = "Enter guardar   Esc volver"

        cv2.putText(
            vista,
            ayuda,
            (px + 18, py + panel_h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (130, 130, 130),
            1,
        )

        cv2.imshow(VENTANA, vista)
        tecla = cv2.waitKeyEx(30)
        tecla_ff = tecla & 0xFF

        if tecla_ff in (27, ord("q")):
            if modo != "accion":
                modo = "accion"
                modo_nuevo = False
                continue
            return None

        if modo == "accion":
            if tecla_ff == ord("s") and puede_confirmar:
                return resultado.nombre
            if tecla_ff == ord("n") and nombres:
                modo = "gesto"
                continue
            if tecla_ff == ord("g"):
                modo = "nuevo"
                modo_nuevo = True
                nombre_nuevo = ""
                continue
            if tecla_ff == ord("c"):
                return None

            flecha = _tecla_flecha(tecla if tecla > 255 else tecla_ff)
            if flecha == "arriba":
                indice_accion = (indice_accion - 1) % len(items_accion)
            elif flecha == "abajo":
                indice_accion = (indice_accion + 1) % len(items_accion)
            elif tecla_ff in (13, 10):
                cod = items_accion[indice_accion][1]
                if cod == "s" and puede_confirmar:
                    return resultado.nombre
                if cod == "n" and nombres:
                    modo = "gesto"
                elif cod == "g":
                    modo = "nuevo"
                    nombre_nuevo = ""
                elif cod == "c":
                    return None

        elif modo == "gesto":
            flecha = _tecla_flecha(tecla if tecla > 255 else tecla_ff)
            if flecha == "arriba" and nombres:
                indice_gesto = (indice_gesto - 1) % len(nombres)
            elif flecha == "abajo" and nombres:
                indice_gesto = (indice_gesto + 1) % len(nombres)
            elif tecla_ff in (13, 10) and nombres:
                return nombres[indice_gesto]
            elif tecla_ff == 27:
                modo = "accion"

        elif modo == "nuevo":
            if tecla_ff in (8, 127):
                nombre_nuevo = nombre_nuevo[:-1]
            elif 32 <= tecla_ff <= 126 and len(nombre_nuevo) < 28:
                nombre_nuevo += chr(tecla_ff)
            elif tecla_ff in (13, 10):
                limpio = nombre_nuevo.strip().lower()
                if limpio:
                    return limpio
            elif tecla_ff == 27:
                modo = "accion"
                modo_nuevo = False


def aviso_breve(cap, mensaje: str, segundos: float = 1.8) -> None:
    fin = time.time() + segundos
    while time.time() < fin:
        vista = _leer_frame(cap)
        if vista is None:
            return
        h, w = vista.shape[:2]
        vista = _oscurecer(vista, 0.35)
        tam = cv2.getTextSize(mensaje, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
        cv2.putText(
            vista,
            mensaje,
            (w // 2 - tam[0] // 2, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (80, 80, 255),
            2,
        )
        cv2.imshow(VENTANA, vista)
        if cv2.waitKey(30) & 0xFF in (27, ord("q")):
            return
