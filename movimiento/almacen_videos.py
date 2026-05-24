"""Almacenamiento de videos por gesto (carpetas uuid_nombre)."""
import json
from datetime import datetime
from pathlib import Path

from movimiento.config import CARPETA_VIDEOS, VIDEO_FPS
from movimiento.utilidades_pose import (
    CARPETA_REFERENCIAS,
    _nombre_limpio,
    buscar_clave_por_nombre,
    clave_con_uuid,
    extraer_uuid,
    nombre_visible,
)

RAIZ_VIDEOS = Path(__file__).resolve().parent.parent / CARPETA_VIDEOS


def _asegurar_raiz() -> Path:
    RAIZ_VIDEOS.mkdir(parents=True, exist_ok=True)
    return RAIZ_VIDEOS


def clave_para_gesto(nombre: str, clave_existente: str | None = None) -> str:
    if clave_existente:
        return clave_existente
    refs = CARPETA_REFERENCIAS
    if refs.exists():
        for carpeta in refs.iterdir():
            if carpeta.is_dir() and nombre_visible(carpeta.name) == _nombre_limpio(nombre).lower():
                return carpeta.name
    return clave_con_uuid(nombre)


def carpeta_gesto(nombre: str, clave: str | None = None) -> Path:
    clave_final = clave_para_gesto(nombre, clave)
    carpeta = _asegurar_raiz() / clave_final
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta


def listar_biblioteca() -> dict[str, list[Path]]:
    """Devuelve {clave_gesto: [rutas .mp4|.avi]}"""
    biblioteca: dict[str, list[Path]] = {}
    if not RAIZ_VIDEOS.exists():
        return biblioteca
    for carpeta in sorted(RAIZ_VIDEOS.iterdir()):
        if not carpeta.is_dir():
            continue
        videos = sorted(
            list(carpeta.glob("*.mp4"))
            + list(carpeta.glob("*.avi"))
            + list(carpeta.glob("*.mkv"))
        )
        if videos:
            biblioteca[carpeta.name] = videos
    return biblioteca


def resumen_biblioteca() -> list[tuple[str, str, int]]:
    """(clave, nombre_visible, num_videos)"""
    filas = []
    for clave, videos in listar_biblioteca().items():
        filas.append((clave, nombre_visible(clave), len(videos)))
    return sorted(filas, key=lambda x: x[1].lower())


def siguiente_ruta_video(carpeta: Path) -> Path:
    existentes = list(carpeta.glob("video_*.mp4"))
    indice = len(existentes)
    return carpeta / f"video_{indice:03d}.mp4"


def guardar_metadatos(carpeta: Path, archivo: str, frames: int, duracion_s: float) -> None:
    meta_path = carpeta / "metadatos.jsonl"
    entrada = {
        "archivo": archivo,
        "frames": frames,
        "duracion_s": round(duracion_s, 2),
        "fps": VIDEO_FPS,
        "fecha": datetime.now().isoformat(timespec="seconds"),
    }
    with meta_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entrada, ensure_ascii=False) + "\n")


def sincronizar_clave_referencia(nombre: str) -> str:
    """Usa la misma clave uuid que referencias si el gesto ya existe."""
    from movimiento.utilidades_pose import cargar_referencias

    refs = cargar_referencias()
    clave = buscar_clave_por_nombre(refs, nombre)
    return clave_para_gesto(nombre, clave)


def gestos_disponibles_para_grabar() -> list[tuple[str, str, int, int]]:
    """
    Lista unificada: gestos con videos y/o referencias.
    (clave, nombre_visible, num_videos, num_muestras)
    """
    from movimiento.utilidades_pose import cargar_referencias

    refs = cargar_referencias()
    bib = listar_biblioteca()
    claves = set(refs.keys()) | set(bib.keys())
    filas = []
    for clave in claves:
        filas.append(
            (
                clave,
                nombre_visible(clave),
                len(bib.get(clave, [])),
                len(refs.get(clave, [])),
            )
        )
    return sorted(filas, key=lambda x: x[1].lower())
