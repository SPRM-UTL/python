"""Veredicto híbrido: LSTM + similitud sobre secuencias normalizadas."""
import numpy as np

from movimiento.config import UMBRAL_CONFIANZA_LSTM
from movimiento.utilidades_pose import (
    detectar_mejor,
    nombre_visible,
    preparar_indice_deteccion,
)


def predecir_gesto(modelo, referencias: dict, secuencia_30: np.ndarray):
    """
    Devuelve (nombre_visible, confianza 0-1, origen).
    origen: 'lstm+sim' | 'similitud' | 'lstm' | '—'
    """
    if not referencias:
        return "Ninguno (Sin modelo)", 0.0, "—"

    claves = sorted(referencias.keys())
    secuencia = np.asarray(secuencia_30, dtype=np.float32)
    indice = preparar_indice_deteccion(referencias)

    nombre_lstm, conf_lstm = None, 0.0
    if modelo is not None:
        pred = modelo.predict(np.expand_dims(secuencia, axis=0), verbose=0)[0]
        idx = int(np.argmax(pred))
        conf_lstm = float(pred[idx])
        nombre_lstm = nombre_visible(claves[idx])

    nombre_sim, score_sim, umbral_sim = detectar_mejor(indice, secuencia)
    sim_valida = bool(nombre_sim) and score_sim >= umbral_sim

    if nombre_lstm and sim_valida:
        if nombre_lstm == nombre_sim:
            return nombre_lstm, max(conf_lstm, score_sim), "lstm+sim"
        if score_sim >= conf_lstm:
            return nombre_sim, score_sim, "similitud"
        if conf_lstm >= UMBRAL_CONFIANZA_LSTM:
            return nombre_lstm, conf_lstm, "lstm"
        return nombre_sim, score_sim, "similitud"

    if sim_valida:
        return nombre_sim, score_sim, "similitud"
    if nombre_lstm and conf_lstm >= UMBRAL_CONFIANZA_LSTM * 0.85:
        return nombre_lstm, conf_lstm, "lstm"
    if nombre_lstm:
        return nombre_lstm, conf_lstm, "lstm"
    return "Ninguno (Sin modelo)", 0.0, "—"
