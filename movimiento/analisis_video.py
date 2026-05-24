"""Extrae secuencias de entrenamiento desde videos grabados."""
from pathlib import Path

import cv2
import numpy as np

from movimiento.almacen_videos import listar_biblioteca, sincronizar_clave_referencia
from movimiento.captura_pose import SesionCaptura
from movimiento.config import (
    LARGO_SECUENCIA,
    MAX_MUESTRAS_POR_VIDEO,
    MIN_FRAMES_VALIDOS_RATIO,
    UMBRAL_DEDUPE_MUESTRA,
    VENTANA_VIDEO_PASO,
)
from movimiento.utilidades_pose import (
    _nombre_limpio,
    buscar_clave_por_nombre,
    cargar_referencias,
    guardar_referencia,
    nombre_visible,
    preparar_secuencia_captura,
    similitud_secuencias,
)


def extraer_vectores_video(
    ruta: Path,
    sesion: SesionCaptura | None = None,
    mostrar_progreso: bool = False,
) -> tuple[list[np.ndarray], int]:
    """
    Lee un video y devuelve (lista de vectores normalizados por frame, frames_sin_pose).
    """
    propio = sesion is None
    if propio:
        sesion = SesionCaptura()

    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        if propio:
            sesion.cerrar()
        raise OSError(f"No se pudo abrir: {ruta}")

    vectores: list[np.ndarray] = []
    sin_pose = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        _, res_pose, res_manos = sesion.procesar_frame(frame)
        vector = sesion.vector_normalizado(res_pose, res_manos)
        if vector is not None:
            vectores.append(vector)
        else:
            sin_pose += 1
        idx += 1
        if mostrar_progreso and total > 0 and idx % 15 == 0:
            print(f"  [{ruta.name}] {idx}/{total} frames...")

    cap.release()
    if propio:
        sesion.cerrar()
    return vectores, sin_pose


def ventanas_desde_vectores(
    vectores: list[np.ndarray],
    largo: int = LARGO_SECUENCIA,
    paso: int = VENTANA_VIDEO_PASO,
    min_ratio_valido: float = MIN_FRAMES_VALIDOS_RATIO,
) -> list[np.ndarray]:
    """Ventanas deslizantes; cada una se normaliza e interpola a `largo` frames."""
    if len(vectores) < largo:
        if len(vectores) >= 8:
            return [preparar_secuencia_captura(vectores, largo)]
        return []

    secuencias = []
    min_validos = int(largo * min_ratio_valido)
    for inicio in range(0, len(vectores) - largo + 1, paso):
        ventana = vectores[inicio : inicio + largo]
        if len(ventana) < min_validos:
            continue
        try:
            sec = preparar_secuencia_captura(ventana, largo)
            secuencias.append(sec)
        except ValueError:
            continue
        if len(secuencias) >= MAX_MUESTRAS_POR_VIDEO:
            break
    return secuencias


def _es_duplicada(secuencia: np.ndarray, existentes: list[np.ndarray]) -> bool:
    for ref in existentes:
        if similitud_secuencias(ref, secuencia) >= UMBRAL_DEDUPE_MUESTRA:
            return True
    return False


def analizar_video(
    ruta: Path,
    sesion: SesionCaptura | None = None,
) -> tuple[list[np.ndarray], dict]:
    """Secuencias candidatas + estadísticas."""
    vectores, sin_pose = extraer_vectores_video(ruta, sesion)
    secuencias = ventanas_desde_vectores(vectores)
    stats = {
        "archivo": ruta.name,
        "frames_totales": len(vectores) + sin_pose,
        "frames_pose": len(vectores),
        "sin_pose": sin_pose,
        "secuencias": len(secuencias),
    }
    return secuencias, stats


def importar_secuencias(
    nombre_gesto: str,
    secuencias: list[np.ndarray],
    clave: str | None = None,
    deduplicar: bool = True,
) -> tuple[int, int]:
    """
    Guarda secuencias en referencias/.
    Devuelve (guardadas, omitidas_duplicadas).
    """
    referencias = cargar_referencias()
    clave = clave or sincronizar_clave_referencia(nombre_gesto)
    existentes = list(referencias.get(clave, []))
    nuevas_en_sesion: list[np.ndarray] = []

    guardadas = 0
    omitidas = 0

    for sec in secuencias:
        pool = existentes + nuevas_en_sesion
        if deduplicar and pool and _es_duplicada(sec, pool):
            omitidas += 1
            continue
        guardar_referencia(
            nombre=nombre_gesto,
            secuencia=sec.tolist(),
            clave_existente=clave,
        )
        nuevas_en_sesion.append(sec)
        guardadas += 1

    return guardadas, omitidas


def analizar_gesto(
    clave: str,
    sesion: SesionCaptura | None = None,
) -> dict:
    """Analiza todos los videos de un gesto. No guarda aún."""
    biblioteca = listar_biblioteca()
    videos = biblioteca.get(clave, [])
    todas: list[np.ndarray] = []
    detalle = []

    for ruta in videos:
        secs, stats = analizar_video(ruta, sesion)
        todas.extend(secs)
        detalle.append(stats)

    return {
        "clave": clave,
        "videos": len(videos),
        "secuencias": todas,
        "detalle": detalle,
    }


def resolver_clave_gesto(nombre_o_clave: str) -> str | None:
    referencias = cargar_referencias()
    biblioteca = listar_biblioteca()
    if nombre_o_clave in referencias or nombre_o_clave in biblioteca:
        return nombre_o_clave
    clave = buscar_clave_por_nombre(referencias, nombre_o_clave)
    if clave:
        return clave
    limpio = _nombre_limpio(nombre_o_clave).lower()
    for k in biblioteca:
        if nombre_visible(k) == limpio or k.endswith(f"_{limpio}"):
            return k
    return None


def analizar_e_importar_gesto(
    nombre_o_clave: str,
    sesion: SesionCaptura | None = None,
) -> dict:
    clave = resolver_clave_gesto(nombre_o_clave)
    if not clave:
        return {"error": f"Gesto no encontrado: {nombre_o_clave}"}

    resultado = analizar_gesto(clave, sesion)
    nombre = nombre_visible(clave)
    guardadas, omitidas = importar_secuencias(
        nombre, resultado["secuencias"], clave=clave
    )
    resultado["guardadas"] = guardadas
    resultado["omitidas"] = omitidas
    resultado["nombre"] = nombre
    return resultado
