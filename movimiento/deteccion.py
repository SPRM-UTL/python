"""Veredicto híbrido: LSTM + similitud sobre secuencias normalizadas."""
from typing import NamedTuple

import numpy as np

from movimiento.config import (
    MARGEN_MINIMO_DETECCION,
    REQUIERE_ACUERDO_LSTM_SIM,
    UMBRAL_CONFIANZA_LSTM,
)
from movimiento.utilidades_pose import (
    detectar_top_k,
    nombre_visible,
    normalizar_secuencia,
    preparar_indice_deteccion,
)


class ResultadoGesto(NamedTuple):
    nombre: str
    confianza: float
    origen: str
    confiable: bool
    nombre_lstm: str | None
    conf_lstm: float
    nombre_sim: str | None
    score_sim: float
    umbral_sim: float
    margen: float
    segunda_opcion: str | None


def _sin_modelo() -> ResultadoGesto:
    return ResultadoGesto(
        "Ninguno (Sin modelo)",
        0.0,
        "—",
        False,
        None,
        0.0,
        None,
        0.0,
        0.0,
        0.0,
        None,
    )


def predecir_gesto(modelo, referencias: dict, secuencia_30: np.ndarray) -> ResultadoGesto:
    """
    Predicción con secuencia ya normalizada o cruda (se normaliza aquí).
    confiable=True solo si LSTM y similitud coinciden, superan umbrales y hay margen.
    """
    if not referencias:
        return _sin_modelo()

    claves = sorted(referencias.keys())
    secuencia = normalizar_secuencia(np.asarray(secuencia_30, dtype=np.float32))
    indice = preparar_indice_deteccion(referencias)

    nombre_lstm, conf_lstm = None, 0.0
    if modelo is not None:
        pred = modelo.predict(np.expand_dims(secuencia, axis=0), verbose=0)[0]
        idx = int(np.argmax(pred))
        conf_lstm = float(pred[idx])
        nombre_lstm = nombre_visible(claves[idx])

    ranking = detectar_top_k(indice, secuencia, k=2)
    if not ranking:
        if nombre_lstm and conf_lstm >= UMBRAL_CONFIANZA_LSTM:
            return ResultadoGesto(
                nombre_lstm,
                conf_lstm,
                "lstm",
                False,
                nombre_lstm,
                conf_lstm,
                None,
                0.0,
                0.0,
                0.0,
                None,
            )
        return _sin_modelo()

    nombre_sim, score_sim, umbral_sim = ranking[0]
    segunda = ranking[1][0] if len(ranking) > 1 else None
    margen = ranking[0][1] - ranking[1][1] if len(ranking) > 1 else ranking[0][1]

    sim_valida = bool(nombre_sim) and score_sim >= umbral_sim
    lstm_valida = bool(nombre_lstm) and conf_lstm >= UMBRAL_CONFIANZA_LSTM
    acuerdo = nombre_lstm == nombre_sim if (nombre_lstm and nombre_sim) else False
    margen_ok = margen >= MARGEN_MINIMO_DETECCION

    confiable = (
        sim_valida
        and lstm_valida
        and acuerdo
        and margen_ok
        and (not REQUIERE_ACUERDO_LSTM_SIM or acuerdo)
    )

    if confiable:
        conf = max(conf_lstm, score_sim)
        return ResultadoGesto(
            nombre_lstm,
            conf,
            "lstm+sim",
            True,
            nombre_lstm,
            conf_lstm,
            nombre_sim,
            score_sim,
            umbral_sim,
            margen,
            segunda,
        )

    if sim_valida and lstm_valida and not acuerdo:
        if score_sim >= conf_lstm and margen_ok:
            return ResultadoGesto(
                nombre_sim,
                score_sim,
                "similitud",
                False,
                nombre_lstm,
                conf_lstm,
                nombre_sim,
                score_sim,
                umbral_sim,
                margen,
                segunda,
            )
        if conf_lstm >= UMBRAL_CONFIANZA_LSTM and margen_ok:
            return ResultadoGesto(
                nombre_lstm,
                conf_lstm,
                "lstm",
                False,
                nombre_lstm,
                conf_lstm,
                nombre_sim,
                score_sim,
                umbral_sim,
                margen,
                segunda,
            )
        return ResultadoGesto(
            "Incertidumbre (sin acuerdo)",
            max(conf_lstm, score_sim),
            "—",
            False,
            nombre_lstm,
            conf_lstm,
            nombre_sim,
            score_sim,
            umbral_sim,
            margen,
            segunda,
        )

    if sim_valida and margen_ok:
        return ResultadoGesto(
            nombre_sim,
            score_sim,
            "similitud",
            False,
            nombre_lstm,
            conf_lstm,
            nombre_sim,
            score_sim,
            umbral_sim,
            margen,
            segunda,
        )

    if lstm_valida:
        return ResultadoGesto(
            nombre_lstm,
            conf_lstm,
            "lstm",
            False,
            nombre_lstm,
            conf_lstm,
            nombre_sim,
            score_sim,
            umbral_sim,
            margen,
            segunda,
        )

    return ResultadoGesto(
        "Incertidumbre (baja confianza)",
        max(conf_lstm, score_sim),
        "—",
        False,
        nombre_lstm,
        conf_lstm,
        nombre_sim if sim_valida else None,
        score_sim,
        umbral_sim,
        margen,
        segunda,
    )
