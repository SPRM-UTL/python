"""Extracción, almacenamiento y comparación de secuencias de movimiento."""
from pathlib import Path

import numpy as np

NUM_LANDMARKS = 33
COORDS_POR_LANDMARK = 3
DIM_FRAME = NUM_LANDMARKS * COORDS_POR_LANDMARK

LANDMARKS_PRIORITARIOS = (11, 12, 13, 14, 15, 16, 23, 24)

CARPETA_REFERENCIAS = Path(__file__).resolve().parent.parent / "referencias"

_cache_pesos: np.ndarray | None = None


def obtener_pesos_frame() -> np.ndarray:
    global _cache_pesos
    if _cache_pesos is None:
        pesos = np.ones(DIM_FRAME, dtype=np.float32)
        for idx in LANDMARKS_PRIORITARIOS:
            pesos[idx * 3 : idx * 3 + 3] = 2.5
        _cache_pesos = pesos
    return _cache_pesos


def landmarks_a_vector(landmarks) -> list[float]:
    vector = []
    for lm in landmarks.landmark:
        vector.extend([lm.x, lm.y, lm.z])
    return vector


def vector_a_matriz(vector) -> np.ndarray:
    return np.array(vector, dtype=np.float32).reshape(NUM_LANDMARKS, COORDS_POR_LANDMARK)


def normalizar_frame(vector: np.ndarray) -> np.ndarray:
    puntos = vector_a_matriz(vector)
    cadera_izq, cadera_der = puntos[23], puntos[24]
    centro = (cadera_izq + cadera_der) / 2.0
    puntos = puntos - centro

    hombro_izq, hombro_der = puntos[11], puntos[12]
    escala = np.linalg.norm(hombro_der[:2] - hombro_izq[:2])
    if escala > 1e-4:
        puntos = puntos / escala
    return puntos.reshape(-1)


def normalizar_secuencia(secuencia: np.ndarray) -> np.ndarray:
    return np.array([normalizar_frame(f) for f in secuencia], dtype=np.float32)


def _similitud_alineada(ref: np.ndarray, act: np.ndarray, max_desfase: int = 6) -> float:
    mejor = 0.0
    pesos = obtener_pesos_frame()

    for desfase in range(-max_desfase, max_desfase + 1):
        if desfase < 0:
            r, a = ref[-desfase:], act[: len(ref) + desfase]
        elif desfase > 0:
            r, a = ref[: len(ref) - desfase], act[desfase:]
        else:
            r, a = ref, act

        if len(r) < 12:
            continue

        diff = (r - a) * pesos
        mse = float(np.mean(diff**2))
        score_mse = 1.0 / (1.0 + mse * 18.0)

        cosenos = []
        for i in range(len(r)):
            ri, ai = r[i], a[i]
            nr, na = float(np.linalg.norm(ri)), float(np.linalg.norm(ai))
            if nr > 1e-6 and na > 1e-6:
                cosenos.append(float(np.dot(ri, ai) / (nr * na)))
        score_cos = float(np.mean(cosenos)) if cosenos else 0.0
        mejor = max(mejor, score_mse, score_cos)

    return mejor


def similitud_secuencias(referencia: np.ndarray, actual: np.ndarray) -> float:
    ref = normalizar_secuencia(referencia)
    act = normalizar_secuencia(actual)
    if ref.shape != act.shape:
        return 0.0
    return _similitud_alineada(ref, act)


def prototipo_muestras(muestras: list[np.ndarray]) -> np.ndarray:
    normalizadas = [normalizar_secuencia(m) for m in muestras]
    return np.mean(normalizadas, axis=0).astype(np.float32)


def mejor_similitud(muestras: list[np.ndarray], actual: np.ndarray) -> float:
    if not muestras:
        return 0.0

    scores = [similitud_secuencias(m, actual) for m in muestras]
    if len(muestras) >= 2:
        proto = prototipo_muestras(muestras)
        scores.append(_similitud_alineada(proto, normalizar_secuencia(actual)))
    return float(max(scores))


def umbral_deteccion(num_muestras: int) -> float:
    return min(0.55, 0.22 + 0.035 * max(1, num_muestras))


def _nombre_limpio(nombre: str) -> str:
    limpio = "".join(c if c.isalnum() or c in "-_" else "_" for c in nombre.strip())
    return limpio or "movimiento"


def _migrar_archivo_plano(archivo: Path) -> None:
    nombre = _nombre_limpio(archivo.stem).lower()
    carpeta = CARPETA_REFERENCIAS / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    destino = carpeta / "muestra_000.npy"
    if not destino.exists():
        datos = np.load(archivo)
        np.save(destino, datos)
    archivo.unlink()


def cargar_referencias() -> dict[str, list[np.ndarray]]:
    referencias: dict[str, list[np.ndarray]] = {}
    if not CARPETA_REFERENCIAS.exists():
        return referencias

    for archivo in list(CARPETA_REFERENCIAS.glob("*.npy")):
        _migrar_archivo_plano(archivo)

    for carpeta in sorted(CARPETA_REFERENCIAS.iterdir()):
        if not carpeta.is_dir():
            continue
        muestras = []
        for archivo in sorted(carpeta.glob("muestra_*.npy")):
            datos = np.load(archivo)
            if len(datos) > 0:
                muestras.append(datos.astype(np.float32))
        if muestras:
            referencias[carpeta.name] = muestras

    return referencias


def guardar_referencia(nombre: str, secuencia: list) -> tuple[Path, int]:
    CARPETA_REFERENCIAS.mkdir(parents=True, exist_ok=True)
    nombre_limpio = _nombre_limpio(nombre).lower()
    carpeta = CARPETA_REFERENCIAS / nombre_limpio
    carpeta.mkdir(parents=True, exist_ok=True)

    existentes = list(carpeta.glob("muestra_*.npy"))
    indice = len(existentes)
    ruta = carpeta / f"muestra_{indice:03d}.npy"
    np.save(ruta, np.array(secuencia, dtype=np.float32))
    return ruta, indice + 1


def resumen_referencias(referencias: dict[str, list[np.ndarray]]) -> str:
    if not referencias:
        return "(ninguna)"
    partes = [f"{n} ({len(m)} muestras)" for n, m in referencias.items()]
    return ", ".join(partes)
