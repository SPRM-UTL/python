import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard

# --- CONFIGURACIÓN ---
DATASET_PATH = "./dataset"
ACCIONES = np.array(["aplaudir", "saludar", "reposo"])
LARGO_SECUENCIA = 30 # Los 30 frames que grabamos por muestra
NUM_LANDMARKS = 99   # 33 puntos x 3 coordenadas (X, Y, Z)

# Mapeo de texto a números (ej: "aplaudir" -> 0, "saludar" -> 1)
label_map = {label: num for num, label in enumerate(ACCIONES)}

secuencias, etiquetas = [], []

print("📂 Cargando dataset desde el disco...")
# 1. Recolectar y organizar los datos
for accion in ACCIONES:
    carpeta_accion = os.path.join(DATASET_PATH, accion)
    archivos = [f for f in os.listdir(carpeta_accion) if f.endswith('.npy')]
    
    print(f" -> Encontradas {len(archivos)} muestras para '{accion}'")
    
    for archivo in archivos:
        ventana_frames = np.load(os.path.join(carpeta_accion, archivo))
        secuencias.append(ventana_frames)
        etiquetas.append(label_map[accion])

# Convertir a arreglos de NumPy
X = np.array(secuencias)
y = to_categorical(etiquetas).astype(int) # Convierte [0, 1, 2] a formato binario (One-Hot)

# Dividir en 80% entrenamiento y 20% pruebas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n📊 Datos listos para entrenamiento:")
print(f" - Set de entrenamiento: {X_train.shape}")
print(f" - Set de pruebas: {X_test.shape}")

# 2. Definir la Arquitectura de la Red Neuronal LSTM
model = Sequential()
# Primera capa LSTM (recibe secuencias de 30 frames, cada uno con 99 coordenadas)
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(LARGO_SECUENCIA, NUM_LANDMARKS)))
model.add(Dropout(0.2)) # Evita que la IA memorice de forma rígida (Overfitting)

# Segunda capa LSTM (procesa la secuencia refinada)
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dropout(0.2))

# Capas densas para clasificar
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
# Capa de salida: 3 neuronas (una por cada acción), usando 'softmax' para darnos porcentajes de probabilidad
model.add(Dense(ACCIONES.shape[0], activation='softmax'))

# Compilar el modelo
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("\n🚀 Iniciando entrenamiento de la IA...")
# 3. Entrenar el modelo
# Modifica epochs=100 si tienes pocas muestras, o súbelo a 200 si quieres más precisión.
model.fit(X_train, y_train, epochs=150, batch_size=4, validation_data=(X_test, y_test))

print("\n📝 Estructura final de tu IA:")
model.summary()

# 4. Guardar el cerebro entrenado
model.save('detector_movimientos.h5')
print("\n✅ ¡Felicidades! Tu IA de movimiento se ha guardado como 'detector_movimientos.h5'")