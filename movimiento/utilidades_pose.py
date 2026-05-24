"""Extracción, almacenamiento y comparación de secuencias de movimiento."""
import math
import uuid
from pathlib import Path

import numpy as np

NUM_LANDMARKS = 33
COORDS_POR_LANDMARK = 3
DIM_POSE = NUM_LANDMARKS * COORDS_POR_LANDMARK
DIM_FRAME = DIM_POSE
NUM_EXTRAS = 6
DIM_FRAME_COMPLETO = DIM_POSE + NUM_EXTRAS

LANDMARKS_PRIORITARIOS = (11, 12, 13, 14, 15, 16, 23, 24)
_PIP_DEDOS = (6, 10, 14, 18)
_PUNTAS_DEDOS = (8, 12, 16, 20)

CARPETA_REFERENCIAS = Path(__file__).resolve().parent.parent / "referencias"

_cache_pesos: np.ndarray | None = None


def obtener_pesos_frame(dim: int = DIM_FRAME_COMPLETO) -> np.ndarray:
    global _cache_pesos
    if _cache_pesos is None or _cache_pesos.shape[0] != dim:
        pesos = np.ones(dim, dtype=np.float32)
        for idx in LANDMARKS_PRIORITARIOS:
            pesos[idx * 3 : idx * 3 + 3] = 2.5
        for i in range(DIM_POSE, dim):
            pesos[i] = 4.0
        _cache_pesos = pesos
    return _cache_pesos


def landmarks_a_vector(landmarks) -> list[float]:
    vector = []
    for lm in landmarks.landmark:
        vector.extend([lm.x, lm.y, lm.z])
    return vector


def _contar_dedos_mano(hand_landmarks, es_izquierda: bool) -> int:
    lm = hand_landmarks.landmark
    extendidos = 0
    if es_izquierda:
        if lm[4].x > lm[3].x:
            extendidos += 1
    elif lm[4].x < lm[3].x:
        extendidos += 1
    for tip, pip in zip(_PUNTAS_DEDOS, _PIP_DEDOS):
        if lm[tip].y < lm[pip].y:
            extendidos += 1
    return extendidos


def _angulo_grados(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dy, dx))


def _inclinacion_desde_pose(lm) -> tuple[float, float, float, float]:
    hombro_izq, hombro_der = lm[11], lm[12]
    cadera_izq, cadera_der = lm[23], lm[24]

    cx_h = (cadera_izq.x + cadera_der.x) / 2.0
    cy_h = (cadera_izq.y + cadera_der.y) / 2.0
    cx_s = (hombro_izq.x + hombro_der.x) / 2.0
    cy_s = (hombro_izq.y + hombro_der.y) / 2.0
    incl_cuerpo = _angulo_grados(cx_s - cx_h, -(cy_s - cy_h))

    incl_hombros = _angulo_grados(
        hombro_der.x - hombro_izq.x,
        hombro_der.y - hombro_izq.y,
    )

    oreja_izq, oreja_der = lm[7], lm[8]
    if oreja_izq.visibility > 0.35 and oreja_der.visibility > 0.35:
        incl_cabeza = _angulo_grados(
            oreja_der.x - oreja_izq.x,
            oreja_der.y - oreja_izq.y,
        )
    else:
        incl_cabeza = 0.0

    brazos = []
    for codo, muneca in ((13, 15), (14, 16)):
        if lm[muneca].visibility > 0.4 and lm[codo].visibility > 0.4:
            brazos.append(
                _angulo_grados(
                    lm[muneca].x - lm[codo].x,
                    lm[muneca].y - lm[codo].y,
                )
            )
    incl_brazo = float(np.mean(brazos)) if brazos else 0.0
    return incl_cuerpo, incl_hombros, incl_cabeza, incl_brazo


def extraer_extras(pose_landmarks, resultados_manos) -> list[float]:
    lm = pose_landmarks.landmark
    dedos_izq, dedos_der = -1.0, -1.0

    if resultados_manos and resultados_manos.multi_hand_landmarks:
        for hand_lm, handedness in zip(
            resultados_manos.multi_hand_landmarks,
            resultados_manos.multi_handedness,
        ):
            etiqueta = handedness.classification[0].label
            cuenta = _contar_dedos_mano(hand_lm, etiqueta == "Left")
            if etiqueta == "Left":
                dedos_izq = float(cuenta)
            else:
                dedos_der = float(cuenta)

    incl_cuerpo, incl_hombros, incl_cabeza, incl_brazo = _inclinacion_desde_pose(lm)
    return [dedos_izq, dedos_der, incl_cuerpo, incl_hombros, incl_cabeza, incl_brazo]


def frame_a_vector_completo(pose_landmarks, resultados_manos) -> list[float]:
    return landmarks_a_vector(pose_landmarks) + extraer_extras(pose_landmarks, resultados_manos)


def resumen_extras(extras: list[float]) -> dict[str, float]:
    return {
        "dedos_izq": extras[0],
        "dedos_der": extras[1],
        "incl_cuerpo": extras[2],
        "incl_hombros": extras[3],
        "incl_cabeza": extras[4],
        "incl_brazo": extras[5],
    }


def completar_frame_legacy(frame: np.ndarray) -> np.ndarray:
    if frame.shape[-1] >= DIM_FRAME_COMPLETO:
        return frame.astype(np.float32)
    if frame.shape[-1] != DIM_POSE:
        return frame.astype(np.float32)
    neutro = np.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.concatenate([frame.astype(np.float32), neutro])


def vector_a_matriz(vector) -> np.ndarray:
    return np.array(vector[:DIM_POSE], dtype=np.float32).reshape(
        NUM_LANDMARKS, COORDS_POR_LANDMARK
    )


def _normalizar_extras(extras: np.ndarray) -> np.ndarray:
    out = extras.astype(np.float32).copy()
    for i in range(2):
        if out[i] < 0:
            out[i] = 0.5
        else:
            out[i] /= 5.0
    out[2:] = np.clip(out[2:] / 90.0, -1.0, 1.0)
    return out


def normalizar_frame(vector: np.ndarray) -> np.ndarray:
    vector = completar_frame_legacy(np.asarray(vector, dtype=np.float32))
    puntos = vector_a_matriz(vector)
    cadera_izq, cadera_der = puntos[23], puntos[24]
    centro = (cadera_izq + cadera_der) / 2.0
    puntos = puntos - centro

    hombro_izq, hombro_der = puntos[11], puntos[12]
    escala = np.linalg.norm(hombro_der[:2] - hombro_izq[:2])
    if escala > 1e-4:
        puntos = puntos / escala
    pose_norm = puntos.reshape(-1)
    extras_norm = _normalizar_extras(vector[DIM_POSE:DIM_FRAME_COMPLETO])
    return np.concatenate([pose_norm, extras_norm])


def normalizar_secuencia(secuencia: np.ndarray) -> np.ndarray:
    return np.array([normalizar_frame(f) for f in secuencia], dtype=np.float32)


def _similitud_alineada(ref: np.ndarray, act: np.ndarray, max_desfase: int = 4) -> float:
    mejor = 0.0
    dim = ref.shape[1] if ref.ndim > 1 else DIM_FRAME_COMPLETO
    pesos = obtener_pesos_frame(dim)

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

        nr = np.linalg.norm(r, axis=1)
        na = np.linalg.norm(a, axis=1)
        validos = (nr > 1e-6) & (na > 1e-6)
        if validos.any():
            cosenos = np.sum(r[validos] * a[validos], axis=1) / (nr[validos] * na[validos])
            score_cos = float(np.mean(cosenos))
        else:
            score_cos = 0.0
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
    act = normalizar_secuencia(actual)
    if len(muestras) >= 2:
        ref = prototipo_muestras(muestras)
    else:
        ref = normalizar_secuencia(muestras[0])
    return _similitud_alineada(ref, act)


def preparar_indice_deteccion(
    referencias: dict[str, list[np.ndarray]],
) -> list[tuple[str, np.ndarray, float]]:
    """Precalcula prototipos normalizados y umbrales (nombre, ref, umbral)."""
    indice = []
    for clave, muestras in referencias.items():
        if not muestras:
            continue
        if len(muestras) >= 2:
            ref = prototipo_muestras(muestras)
        else:
            ref = normalizar_secuencia(muestras[0])
        indice.append((nombre_visible(clave), ref, umbral_deteccion(len(muestras))))
    return indice


def detectar_mejor(
    indice: list[tuple[str, np.ndarray, float]],
    actual: np.ndarray,
) -> tuple[str, float, float]:
    ranking = detectar_top_k(indice, actual, k=1)
    if not ranking:
        return "", 0.0, 0.0
    nombre, score, umbral = ranking[0]
    return nombre, score, umbral


def detectar_top_k(
    indice: list[tuple[str, np.ndarray, float]],
    actual: np.ndarray,
    k: int = 2,
) -> list[tuple[str, float, float]]:
    act = normalizar_secuencia(actual)
    resultados: list[tuple[str, float, float]] = []
    for nombre, ref, umbral in indice:
        score = _similitud_alineada(ref, act)
        resultados.append((nombre, score, umbral))
    resultados.sort(key=lambda x: x[1], reverse=True)
    return resultados[: max(1, k)]


def ajustar_secuencia_a_largo(
    secuencia: np.ndarray,
    largo_objetivo: int,
    *,
    normalizar: bool = True,
) -> np.ndarray:
    """Interpola temporalmente a largo fijo; opcionalmente normaliza cada frame."""
    arr = np.asarray(secuencia, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    largo_actual = len(arr)
    if largo_actual == 0:
        raise ValueError("La secuencia está vacía")

    if largo_actual != largo_objetivo:
        indices_originales = np.linspace(0, largo_actual - 1, num=largo_actual)
        indices_nuevos = np.linspace(0, largo_actual - 1, num=largo_objetivo)
        ajustada = np.zeros((largo_objetivo, arr.shape[1]), dtype=np.float32)
        for coord in range(arr.shape[1]):
            ajustada[:, coord] = np.interp(
                indices_nuevos, indices_originales, arr[:, coord]
            )
        arr = ajustada

    if normalizar:
        arr = normalizar_secuencia(arr)
    return arr.astype(np.float32)


def preparar_secuencia_captura(
    frames: list,
    largo: int = 30,
) -> np.ndarray:
    """Pipeline único: interpolar a N frames y normalizar espacialmente."""
    return ajustar_secuencia_a_largo(np.array(frames, dtype=np.float32), largo, normalizar=True)


def umbral_deteccion(num_muestras: int) -> float:
    return min(0.62, 0.30 + 0.028 * max(1, num_muestras))


def _nombre_limpio(nombre: str) -> str:
    limpio = "".join(c if c.isalnum() or c in "-_" else "_" for c in nombre.strip())
    return limpio or "movimiento"


def _es_prefijo_uuid(clave: str) -> bool:
    if len(clave) <= 37 or clave[36] != "_":
        return False
    try:
        uuid.UUID(clave[:36])
        return True
    except ValueError:
        return False


def extraer_uuid(clave: str) -> str | None:
    if _es_prefijo_uuid(clave):
        return clave[:36]
    return None


def nombre_visible(clave: str) -> str:
    if _es_prefijo_uuid(clave):
        return clave[37:]
    return clave


def clave_con_uuid(nombre: str, uuid_str: str | None = None) -> str:
    nombre_limpio = _nombre_limpio(nombre).lower()
    uid = uuid_str or str(uuid.uuid4())
    return f"{uid}_{nombre_limpio}"


def buscar_clave_por_nombre(referencias: dict, nombre: str) -> str | None:
    nombre_limpio = _nombre_limpio(nombre).lower()
    for clave in referencias:
        if nombre_visible(clave) == nombre_limpio or clave == nombre_limpio:
            return clave
    return None


def _migrar_carpeta_a_uuid(clave: str, nombre: str) -> str:
    if extraer_uuid(clave):
        return clave
    nueva_clave = clave_con_uuid(nombre)
    carpeta_vieja = CARPETA_REFERENCIAS / clave
    carpeta_nueva = CARPETA_REFERENCIAS / nueva_clave
    if carpeta_vieja.exists() and not carpeta_nueva.exists():
        carpeta_vieja.rename(carpeta_nueva)
    return nueva_clave


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
            datos = np.load(archivo).astype(np.float32)
            if len(datos) == 0:
                continue
            if datos.ndim == 1:
                datos = datos.reshape(1, -1)
            muestras.append(
                np.array([completar_frame_legacy(f) for f in datos], dtype=np.float32)
            )
        if muestras:
            referencias[carpeta.name] = muestras

    return referencias


def guardar_referencia(
    nombre: str,
    secuencia: list,
    clave_existente: str | None = None,
) -> tuple[Path, int, str]:
    CARPETA_REFERENCIAS.mkdir(parents=True, exist_ok=True)

    if clave_existente:
        clave = _migrar_carpeta_a_uuid(clave_existente, nombre)
    else:
        clave = clave_con_uuid(nombre)

    carpeta = CARPETA_REFERENCIAS / clave
    carpeta.mkdir(parents=True, exist_ok=True)

    existentes = list(carpeta.glob("muestra_*.npy"))
    indice = len(existentes)
    ruta = carpeta / f"muestra_{indice:03d}.npy"
    np.save(ruta, np.array(secuencia, dtype=np.float32))
    return ruta, indice + 1, clave


def resumen_referencias(referencias: dict[str, list[np.ndarray]]) -> str:
    if not referencias:
        return "(ninguna)"
    partes = [f"{nombre_visible(n)} ({len(m)} muestras)" for n, m in referencias.items()]
    return ", ".join(partes)
