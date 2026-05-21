"""Interfaz OpenCV: menu, grabacion de video y analisis."""
import time
from pathlib import Path

import cv2
import numpy as np

from movimiento.almacen_videos import (
    carpeta_gesto,
    gestos_disponibles_para_grabar,
    guardar_metadatos,
    resumen_biblioteca,
    siguiente_ruta_video,
    sincronizar_clave_referencia,
)
from movimiento.analisis_video import analizar_e_importar_gesto, analizar_gesto
from movimiento.config import (
    FRAMES_VIDEO_AUTO,
    LARGO_SECUENCIA,
    MIN_FRAMES_VIDEO,
    VENTANA_VIDEOS,
    VIDEO_FPS,
)
from movimiento.ui_camara import (
    _leer_frame,
    _oscurecer,
    _tecla_flecha,
    aviso_breve,
    cuenta_regresiva,
    hud_procesando,
    pedir_nombre_movimiento,
)

MENU_OPCIONES = [
    ("Grabar video", "1"),
    ("Biblioteca de videos", "2"),
    ("Analizar videos -> IA", "3"),
    ("Entrenar IA", "4"),
    ("Salir", "q"),
]


def _dibujar_panel(
    vista: np.ndarray,
    titulo: str,
    lineas: list[str],
    items: list[tuple[str, str]] | None = None,
    indice_sel: int = 0,
    ayuda: str = "",
) -> None:
    h, w = vista.shape[:2]
    vista[:] = _oscurecer(vista, 0.45)
    panel_w = min(640, w - 40)
    n_items = len(items) if items else 0
    panel_h = 100 + len(lineas) * 24 + n_items * 36 + (40 if ayuda else 0)
    px, py = (w - panel_w) // 2, max(24, (h - panel_h) // 2)

    cv2.rectangle(vista, (px, py), (px + panel_w, py + panel_h), (38, 38, 38), -1)
    cv2.rectangle(vista, (px, py), (px + panel_w, py + panel_h), (0, 200, 255), 2)
    cv2.putText(
        vista, titulo, (px + 18, py + 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2, cv2.LINE_AA,
    )
    y = py + 58
    for ln in lineas:
        cv2.putText(
            vista, ln[:56], (px + 18, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1, cv2.LINE_AA,
        )
        y += 24

    if items:
        y += 8
        for i, (etiq, tecla) in enumerate(items):
            sel = i == indice_sel
            color = (120, 255, 120) if sel else (200, 200, 200)
            pref = "> " if sel else "  "
            cv2.putText(
                vista, f"{pref}[{tecla}] {etiq}"[:52], (px + 20, y + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2 if sel else 1, cv2.LINE_AA,
            )
            y += 36

    if ayuda:
        cv2.putText(
            vista, ayuda, (px + 18, py + panel_h - 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1, cv2.LINE_AA,
        )


def pedir_opcion_menu(cap) -> str | None:
    indice = 0
    while True:
        vista = _leer_frame(cap)
        if vista is None:
            return None
        _dibujar_panel(
            vista,
            "BIBLIOTECA DE VIDEOS",
            [
                "Graba gestos en video y analizalos despues",
                "para generar muchas muestras de entrenamiento.",
            ],
            MENU_OPCIONES,
            indice,
            "Flechas / W-S   Enter   Esc",
        )
        cv2.imshow(VENTANA_VIDEOS, vista)
        tecla = cv2.waitKeyEx(30)
        ff = tecla & 0xFF
        if ff in (27, ord("q")):
            return None
        if ff in (13, 10):
            return MENU_OPCIONES[indice][1]
        flecha = _tecla_flecha(tecla if tecla > 255 else ff)
        if flecha == "arriba":
            indice = (indice - 1) % len(MENU_OPCIONES)
        elif flecha == "abajo":
            indice = (indice + 1) % len(MENU_OPCIONES)
        for i, (_, cod) in enumerate(MENU_OPCIONES):
            if ff == ord(cod):
                return cod


def pedir_gesto_lista(
    cap,
    titulo: str = "Elegir gesto",
    permitir_nuevo: bool = True,
) -> tuple[str, str] | None:
    """
    Devuelve (clave, nombre_visible) o None.
    """
    filas = gestos_disponibles_para_grabar()
    items: list[tuple[str, str | None]] = []
    if permitir_nuevo:
        items.append(("+ Nuevo gesto", None))
    for clave, nombre, nv, nm in filas:
        items.append((f"{nombre}  (v:{nv}  m:{nm})", clave))
    if len(items) == 1 and permitir_nuevo:
        nombre = pedir_nombre_movimiento(cap, [])
        if not nombre:
            return None
        clave = sincronizar_clave_referencia(nombre)
        return clave, nombre.strip().lower()

    indice = 0
    modo_nuevo = False
    nombre_nuevo = ""

    while True:
        vista = _leer_frame(cap)
        if vista is None:
            return None

        if modo_nuevo:
            _dibujar_panel(
                vista, titulo, ["Nombre del gesto:"],
                ayuda="Escribir + Enter   Esc volver",
            )
            h, w = vista.shape[:2]
            cv2.putText(
                vista, nombre_nuevo or "ej: aplaudir", (w // 2 - 120, h // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )
        else:
            lista_ui = [(items[i][0], str(i)) for i in range(len(items))]
            _dibujar_panel(
                vista, titulo,
                [f"{len(filas)} gestos en el sistema"],
                lista_ui, indice,
                "Flechas   Enter   Esc",
            )

        cv2.imshow(VENTANA_VIDEOS, vista)
        tecla = cv2.waitKeyEx(30)
        ff = tecla & 0xFF

        if ff in (27, ord("q")):
            if modo_nuevo:
                modo_nuevo = False
                continue
            return None

        if modo_nuevo:
            if ff in (8, 127):
                nombre_nuevo = nombre_nuevo[:-1]
            elif 32 <= ff <= 126 and len(nombre_nuevo) < 28:
                nombre_nuevo += chr(ff)
            elif ff in (13, 10) and nombre_nuevo.strip():
                n = nombre_nuevo.strip().lower()
                return sincronizar_clave_referencia(n), n
            continue

        flecha = _tecla_flecha(tecla if tecla > 255 else ff)
        if flecha == "arriba":
            indice = (indice - 1) % len(items)
        elif flecha == "abajo":
            indice = (indice + 1) % len(items)
        elif ff in (13, 10):
            _, clave = items[indice]
            if clave is None:
                modo_nuevo = True
                nombre_nuevo = ""
            else:
                from movimiento.utilidades_pose import nombre_visible
                return clave, nombre_visible(clave)


def hud_grabando_video(
    vista,
    nombre: str,
    frames_pose: int,
    meta: int = FRAMES_VIDEO_AUTO,
    sin_pose_aviso: bool = False,
) -> None:
    h, w = vista.shape[:2]
    progreso = min(1.0, frames_pose / meta) if meta else 0.0
    cv2.rectangle(vista, (0, 0), (w, h), (0, 0, 220), 8)
    overlay = vista.copy()
    cv2.rectangle(overlay, (0, 0), (w, 62), (40, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, vista, 0.35, 0, vista)
    cv2.putText(
        vista, f"REC — {nombre}", (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 80, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        vista, f"Frame {frames_pose} / {meta}  (auto al completar)", (14, 52),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
    )
    bar_x, bar_y, bar_w = 14, 58, w - 28
    cv2.rectangle(vista, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (50, 50, 50), -1)
    fill = int(bar_w * progreso)
    if fill > 0:
        cv2.rectangle(vista, (bar_x, bar_y), (bar_x + fill, bar_y + 8), (0, 200, 255), -1)
    if sin_pose_aviso:
        cv2.putText(
            vista, "Mantente visible frente a la camara", (14, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 160, 255), 1, cv2.LINE_AA,
        )
    cv2.putText(
        vista, "[Q] Cancelar grabacion", (14, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA,
    )


def _flash_video_completo(cap, nombre: str, archivo: str) -> None:
    fin = time.time() + 1.2
    while time.time() < fin:
        vista = _leer_frame(cap)
        if vista is None:
            return
        h, w = vista.shape[:2]
        vista = _oscurecer(vista, 0.35)
        cv2.putText(
            vista, "VIDEO COMPLETO", (w // 2 - 140, h // 2 - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 255, 120), 2, cv2.LINE_AA,
        )
        cv2.putText(
            vista, f"{nombre} — {FRAMES_VIDEO_AUTO} frames", (w // 2 - 160, h // 2 + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
        )
        cv2.putText(
            vista, archivo[:40], (w // 2 - 120, h // 2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA,
        )
        cv2.imshow(VENTANA_VIDEOS, vista)
        if cv2.waitKey(30) & 0xFF in (27, ord("q")):
            return


def pedir_post_grabacion_video(cap, nombre: str, archivo: str) -> str | None:
    """
    Tras guardar un video.
    Devuelve 'repetir' | 'menu' | None (cancelar).
    """
    items = [
        (f"Grabar otro video de '{nombre}'", "s"),
        ("Continuar (volver al menu)", "n"),
    ]
    indice = 0
    lineas = [
        f"Guardado: {archivo}",
        f"{FRAMES_VIDEO_AUTO} frames con pose detectada",
        "",
        "Quieres repetir el mismo gesto?",
    ]

    while True:
        vista = _leer_frame(cap)
        if vista is None:
            return None
        _dibujar_panel(
            vista,
            "GRABACION TERMINADA",
            lineas,
            items,
            indice,
            "[S] Repetir   [N] Continuar   Flechas   Enter   Esc",
        )
        cv2.imshow(VENTANA_VIDEOS, vista)
        tecla = cv2.waitKeyEx(30)
        ff = tecla & 0xFF
        if ff in (27, ord("q")):
            return None
        if ff == ord("s"):
            return "repetir"
        if ff == ord("n"):
            return "menu"
        if ff in (13, 10):
            return "repetir" if indice == 0 else "menu"
        flecha = _tecla_flecha(tecla if tecla > 255 else ff)
        if flecha == "arriba":
            indice = (indice - 1) % len(items)
        elif flecha == "abajo":
            indice = (indice + 1) % len(items)


def grabar_video_gesto(cap, sesion, clave: str, nombre: str) -> Path | None:
    """Graba MP4; termina sola al alcanzar FRAMES_VIDEO_AUTO frames con pose."""
    from movimiento.config import CUENTA_REGRESIVA_SEG

    if not cuenta_regresiva(cap, nombre, segundos=CUENTA_REGRESIVA_SEG):
        return None

    carpeta = carpeta_gesto(nombre, clave)
    ruta = siguiente_ruta_video(carpeta)
    inicio = time.time()
    frames_pose = 0
    sesion.reiniciar_suavizado()
    writer = None
    sin_pose_reciente = False

    while cap.isOpened() and frames_pose < FRAMES_VIDEO_AUTO:
        ret, frame = cap.read()
        if not ret:
            break
        vista, res_pose, res_manos = sesion.procesar_frame(frame)
        if writer is None:
            h, w = vista.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(ruta), fourcc, VIDEO_FPS, (w, h))
            if not writer.isOpened():
                aviso_breve(cap, "Error al crear el video.")
                return None

        sesion.dibujar_esqueleto(vista, res_pose, res_manos)
        tiene_pose = sesion.vector_normalizado(res_pose, res_manos) is not None
        sin_pose_reciente = not tiene_pose

        if tiene_pose:
            writer.write(vista)
            frames_pose += 1

        hud_grabando_video(vista, nombre, frames_pose, sin_pose_aviso=sin_pose_reciente)
        cv2.imshow(VENTANA_VIDEOS, vista)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            if writer:
                writer.release()
            ruta.unlink(missing_ok=True)
            return None

    if writer:
        writer.release()
    else:
        ruta.unlink(missing_ok=True)
        aviso_breve(cap, "No se grabo ningun frame.")
        return None

    duracion = time.time() - inicio

    if frames_pose < MIN_FRAMES_VIDEO:
        ruta.unlink(missing_ok=True)
        aviso_breve(
            cap,
            f"Solo {frames_pose} frames con pose. Mantente visible e intenta de nuevo.",
        )
        return None

    guardar_metadatos(carpeta, ruta.name, frames_pose, duracion)
    _flash_video_completo(cap, nombre, ruta.name)
    return ruta


def flujo_grabar_videos(cap, sesion) -> None:
    """Elige gesto y permite repetir grabacion del mismo o volver al menu."""
    sel = pedir_gesto_lista(cap, "Grabar video para...")
    if not sel:
        return
    clave, nombre = sel

    while True:
        ruta = grabar_video_gesto(cap, sesion, clave, nombre)
        if not ruta:
            break
        print(f"[Video] Guardado {ruta}")
        accion = pedir_post_grabacion_video(cap, nombre, ruta.name)
        if accion is None or accion == "menu":
            break
        if accion == "repetir":
            continue


def mostrar_biblioteca(cap) -> None:
    filas = resumen_biblioteca()
    gestos = gestos_disponibles_para_grabar()
    lineas = ["Videos por gesto:"]
    if not filas:
        lineas.append("(sin videos — usa [1] Grabar)")
    for _, nombre, nv in filas:
        lineas.append(f"  {nombre}: {nv} video(s)")
    lineas.append("")
    lineas.append("Muestras IA (referencias/):")
    for _, nombre, _, nm in gestos[:8]:
        lineas.append(f"  {nombre}: {nm} muestra(s)")

    while True:
        vista = _leer_frame(cap)
        if vista is None:
            return
        _dibujar_panel(
            vista, "BIBLIOTECA", lineas,
            ayuda="Cualquier tecla volver",
        )
        cv2.imshow(VENTANA_VIDEOS, vista)
        if cv2.waitKey(0) & 0xFF != 255:
            return


def mostrar_resultado_analisis(cap, resultado: dict) -> bool:
    """Muestra resumen. Devuelve True si el usuario quiere entrenar."""
    if resultado.get("error"):
        aviso_breve(cap, resultado["error"])
        return False

    lineas = [
        f"Gesto: {resultado.get('nombre', '?')}",
        f"Videos analizados: {resultado.get('videos', 0)}",
        f"Secuencias detectadas: {len(resultado.get('secuencias', []))}",
        f"Guardadas en referencias: {resultado.get('guardadas', 0)}",
        f"Omitidas (duplicadas): {resultado.get('omitidas', 0)}",
    ]
    for d in resultado.get("detalle", [])[:6]:
        lineas.append(
            f"  {d['archivo']}: {d['frames_pose']}f -> {d['secuencias']} seq"
        )

    items = [("Entrenar IA ahora", "t"), ("Volver al menu", "m")]
    indice = 0

    while True:
        vista = _leer_frame(cap)
        if vista is None:
            return False
        _dibujar_panel(
            vista, "ANALISIS COMPLETADO", lineas, items, indice,
            "Flechas   Enter",
        )
        cv2.imshow(VENTANA_VIDEOS, vista)
        tecla = cv2.waitKeyEx(30)
        ff = tecla & 0xFF
        if ff in (13, 10):
            return items[indice][1] == "t"
        flecha = _tecla_flecha(tecla if tecla > 255 else ff)
        if flecha == "arriba":
            indice = (indice - 1) % 2
        elif flecha == "abajo":
            indice = (indice + 1) % 2
        if ff == ord("t"):
            return True
        if ff in (27, ord("m"), ord("q")):
            return False


def flujo_analizar(cap, sesion) -> bool:
    """Analiza videos de un gesto e importa. Devuelve si debe entrenar."""
    sel = pedir_gesto_lista(cap, "Analizar videos de...", permitir_nuevo=False)
    if not sel:
        return False
    clave, nombre = sel

    from movimiento.almacen_videos import listar_biblioteca
    if clave not in listar_biblioteca():
        aviso_breve(cap, f"Sin videos para '{nombre}'. Graba primero.")
        return False

    vista = _leer_frame(cap)
    if vista is not None:
        hud_procesando(vista, f"Analizando {nombre}...")
        cv2.imshow(VENTANA_VIDEOS, vista)

    print(f"\n[Analisis] Procesando videos de '{nombre}'...")
    resultado = analizar_e_importar_gesto(clave, sesion)
    print(
        f"  Guardadas: {resultado.get('guardadas', 0)}, "
        f"omitidas: {resultado.get('omitidas', 0)}"
    )
    return mostrar_resultado_analisis(cap, resultado)
