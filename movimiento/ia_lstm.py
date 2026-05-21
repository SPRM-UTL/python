"""Red LSTM: entrenamiento y carga sobre referencias normalizadas."""
import os

import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical

from movimiento.config import BATCH_SIZE, EPOCHS_ENTRENAMIENTO, LARGO_SECUENCIA, MODEL_NAME
from movimiento.utilidades_pose import (
    DIM_FRAME_COMPLETO,
    ajustar_secuencia_a_largo,
    cargar_referencias,
    normalizar_secuencia,
)


def descubrir_dimension_vector(referencias) -> int:
    for muestras in referencias.values():
        if muestras and len(muestras[0]) > 0:
            return int(np.array(muestras[0]).shape[-1])
    return DIM_FRAME_COMPLETO


def _preparar_muestra(muestra: np.ndarray, tam_input: int) -> np.ndarray | None:
    arr = np.asarray(muestra, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[-1] != tam_input:
        return None
    if len(arr) != LARGO_SECUENCIA:
        arr = ajustar_secuencia_a_largo(arr, LARGO_SECUENCIA, normalizar=True)
    elif arr.shape == (LARGO_SECUENCIA, tam_input):
        arr = normalizar_secuencia(arr)
    else:
        return None
    if arr.shape != (LARGO_SECUENCIA, tam_input):
        return None
    return arr


def construir_o_adaptar_modelo(num_clases: int, tam_input: int, modelo_anterior=None):
    model = Sequential()
    model.add(
        LSTM(
            64,
            return_sequences=True,
            activation="relu",
            input_shape=(LARGO_SECUENCIA, tam_input),
        )
    )
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(num_clases, activation="softmax"))
    model.compile(
        optimizer="Adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    if modelo_anterior is not None:
        for i in range(len(model.layers) - 1):
            try:
                model.layers[i].set_weights(modelo_anterior.layers[i].get_weights())
            except Exception:
                pass
    return model


def entrenar_desde_referencias(modelo_actual=None):
    referencias = cargar_referencias()
    if not referencias:
        return modelo_actual

    tam_input = descubrir_dimension_vector(referencias)
    claves_ordenadas = sorted(referencias.keys())
    label_map = {clave: num for num, clave in enumerate(claves_ordenadas)}

    secuencias, etiquetas = [], []
    print(f"\n[IA] Re-entrenando con {len(claves_ordenadas)} clases...")

    for clave, muestras in referencias.items():
        for muestra in muestras:
            arr = _preparar_muestra(muestra, tam_input)
            if arr is not None:
                secuencias.append(arr)
                etiquetas.append(label_map[clave])

    if len(secuencias) < 2:
        print("[IA] Datos insuficientes (mínimo 2 muestras en total).")
        return modelo_actual

    X = np.array(secuencias)
    y = to_categorical(etiquetas, num_classes=len(claves_ordenadas)).astype(int)

    if (
        modelo_actual is None
        or modelo_actual.layers[-1].output_shape[-1] != len(claves_ordenadas)
    ):
        modelo_actual = construir_o_adaptar_modelo(
            len(claves_ordenadas), tam_input, modelo_anterior=modelo_actual
        )

    modelo_actual.fit(X, y, epochs=EPOCHS_ENTRENAMIENTO, batch_size=BATCH_SIZE, verbose=1)
    modelo_actual.save(MODEL_NAME)
    print("[IA] Modelo guardado.\n")
    return modelo_actual


def cargar_o_entrenar():
    referencias = cargar_referencias()
    modelo = None
    if os.path.exists(MODEL_NAME):
        print(f"Cargando modelo: {MODEL_NAME}")
        modelo = load_model(MODEL_NAME)
        if referencias and modelo.layers[-1].output_shape[-1] != len(referencias):
            modelo = entrenar_desde_referencias(modelo)
    elif referencias:
        modelo = entrenar_desde_referencias(modelo)
    return modelo
