import os
import sys
import time
import cv2
import numpy as np
import mediapipe as mp

# Silenciar logs innecesarios de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Reutilizar las funciones nativas de tu repositorio
from movimiento.utilidades_pose import (
    cargar_referencias,
    frame_a_vector_completo,
    guardar_referencia,
    nombre_visible,
    buscar_clave_por_nombre
)

import types
try:
    import google.protobuf.runtime_version
except ImportError:
    mod = types.ModuleType('google.protobuf.runtime_version')
    mod.runtime_version = lambda *args, **kwargs: None
    mod.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None
    sys.modules['google.protobuf.runtime_version'] = mod

# --- CONFIGURACIÓN ---
LARGO_SECUENCIA = 30 #
MODEL_NAME = 'ia_detector_referencias.h5'

def descubrir_dimension_vector(referencias):
    for clave, muestras in referencias.items():
        if muestras and len(muestras[0]) > 0:
            return np.array(muestras[0]).shape[-1]
    return 105 

def ajustar_secuencia_a_30(secuencia, largo_objetivo=30):
    """ 
    Toma una lista de frames de cualquier tamaño y la interpola linealmente 
    para que contenga exactamente 30 frames, manteniendo la estructura temporal.
    """
    secuencia_arr = np.array(secuencia, dtype=np.float32)
    largo_actual = len(secuencia_arr)
    
    if largo_actual == largo_objetivo:
        return secuencia_arr
        
    # Crear índices de interpolación
    indices_originales = np.linspace(0, largo_actual - 1, num=largo_actual)
    indices_nuevos = np.linspace(0, largo_actual - 1, num=largo_objetivo)
    
    secuencia_ajustada = np.zeros((largo_objetivo, secuencia_arr.shape[1]), dtype=np.float32)
    
    for coordenada_idx in range(secuencia_arr.shape[1]):
        secuencia_ajustada[:, coordenada_idx] = np.interp(
            indices_nuevos, 
            indices_originales, 
            secuencia_arr[:, coordenada_idx]
        )
        
    return secuencia_ajustada.tolist()

def construir_o_adaptar_modelo(num_clases, tam_input, modelo_anterior=None):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(LARGO_SECUENCIA, tam_input)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_clases, activation='softmax'))
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    if modelo_anterior is not None:
        for i in range(len(model.layers) - 1):
            try:
                model.layers[i].set_weights(modelo_anterior.layers[i].get_weights())
            except Exception:
                pass
    return model

def entrenar_ia_desde_referencias(modelo_actual):
    referencias = cargar_referencias() #
    if not referencias:
        return modelo_actual

    tam_input = descubrir_dimension_vector(referencias)
    claves_ordenadas = sorted(list(referencias.keys()))
    
    # !!! CORRECCIÓN AQUÍ: Asegurar que coincida la variable del diccionario con el ciclo for
    label_map = {clave: num for num, clave in enumerate(claves_ordenadas)}
    
    secuencias, etiquetas = [], []
    print(f"\n[IA] Re-entrenando red evolutiva con {len(claves_ordenadas)} clases...")
    
    for clave, muestras in referencias.items():
        for muestra in muestras:
            arr_muestra = np.array(muestra, dtype=np.float32)
            if arr_muestra.shape == (LARGO_SECUENCIA, tam_input):
                secuencias.append(arr_muestra)
                etiquetas.append(label_map[clave])
                
    if len(secuencias) < 2:
        return modelo_actual

    X = np.array(secuencias)
    y = to_categorical(etiquetas, num_classes=len(claves_ordenadas)).astype(int) #
    
    if modelo_actual is None or modelo_actual.layers[-1].output_shape[-1] != len(claves_ordenadas):
        modelo_actual = construir_o_adaptar_modelo(len(claves_ordenadas), tam_input, modelo_anterior=modelo_actual)

    modelo_actual.fit(X, y, epochs=35, batch_size=4, verbose=1)
    modelo_actual.save(MODEL_NAME) #
    print("[IA] ¡Modelo actualizado!\n")
    return modelo_actual

def main():
    referencias = cargar_referencias() #
    
    modelo = None
    if os.path.exists(MODEL_NAME):
        print(f"Cargando modelo neuronal: {MODEL_NAME}")
        modelo = load_model(MODEL_NAME)
        if referencias and modelo.layers[-1].output_shape[-1] != len(referencias):
            modelo = entrenar_ia_desde_referencias(modelo)
    else:
        if referencias:
            modelo = entrenar_ia_desde_referencias(modelo)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) #

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) #
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) #

    # Lista dinámica para guardar la ráfaga delimitada por el usuario
    buffer_dinamico = [] 
    grabando_movimiento = False
    
    print("\n" + "="*70)
    print("  ASISTENTE POR IA: DELIMITACIÓN DE INICIO Y FIN CON [ESPACIO]")
    print("="*70)
    print(" Flujo:")
    print("  1. Presiona [ESPACIO] para INICIAR la grabación del gesto.")
    print("  2. Realiza el movimiento frente a la cámara.")
    print("  3. Presiona [ESPACIO] otra vez para TERMINAR y procesar.")
    print("  4. Presiona [q] para salir del programa.")
    print("-"*70 + "\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #
        resultados_pose = pose.process(frame_rgb) #
        resultados_manos = hands.process(frame_rgb) #
        vista = cv2.flip(frame, 1) #

        # Si el interruptor de grabación está encendido, acumulamos frames
        if grabando_movimiento:
            estado_texto = f"REC - GRABANDO: {len(buffer_dinamico)} frames acumulados"
            color_texto = (0, 0, 255) # Rojo indicando grabación activa
            
            if resultados_pose.pose_landmarks:
                vector = frame_a_vector_completo(resultados_pose.pose_landmarks, resultados_manos) #
                buffer_dinamico.append(vector)
        else:
            estado_texto = "SISTEMA LISTO (IA DORMIDA)"
            color_texto = (0, 255, 0) # Verde listo

        # UI en pantalla OpenCV
        cv2.putText(vista, estado_texto, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_texto, 2) #
        cv2.putText(vista, "[Espacio] Marcar Inicio/Fin  -  [Q] Salir", (15, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Efecto de borde rojo en la pantalla si se está grabando para dar feedback visual
        if grabando_movimiento:
            cv2.rectangle(vista, (0, 0), (vista.shape[1], vista.shape[0]), (0, 0, 255), 6) #

        cv2.imshow("Asistente Delimitado", vista)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
        if key == ord(' '):
            if not grabando_movimiento:
                # --- GATILLO 1: INICIAR CAPTURA ---
                grabando_movimiento = True
                buffer_dinamico.clear()
                print("[!] Grabación iniciada... Realiza tu movimiento ahora.")
            else:
                # --- GATILLO 2: DETENER Y PROCESAR ---
                grabando_movimiento = False
                
                if len(buffer_dinamico) < 5:
                    print("[!] Movimiento demasiado corto. Inténtalo de nuevo.")
                    buffer_dinamico.clear()
                    continue
                
                print("\n" + "*"*60)
                print(f" ¡CAPTURA COMPLETADA! ({len(buffer_dinamico)} frames originales)")
                print(" Ajustando dinámicamente la muestra a 30 frames para la IA...")
                print("*"*60)
                
                # Normalizar el tamaño de la ráfaga usando interpolación temporal lineal
                secuencia_normalizada = ajustar_secuencia_a_30(buffer_dinamico, largo_objetivo=LARGO_SECUENCIA)
                
                referencias = cargar_referencias()
                claves_ordenadas = sorted(list(referencias.keys()))
                nombre_predicho = "Ninguno (Sin modelo)"
                confianza = 0.0

                # Ejecutar predicción de la IA en la ráfaga normalizada de 30 frames
                if modelo is not None and referencias:
                    entrada_ia = np.expand_dims(np.array(secuencia_normalizada, dtype=np.float32), axis=0)
                    prediccion = modelo.predict(entrada_ia, verbose=0)[0]
                    idx = np.argmax(prediccion)
                    confianza = prediccion[idx]
                    nombre_predicho = nombre_visible(claves_ordenadas[idx]) #

                print(f"\n[IA Veredicto]: Candidato -> '{nombre_predicho.upper()}' ({confianza*100:.1f}% confianza).")
                print("\n¿Qué deseas hacer con este movimiento?")
                print(f"  [s] Confirmar: Sí, es '{nombre_predicho}'")
                print("  [n] Corregir: Fue otra acción de la lista existente")
                print("  [g] Crear: Es un movimiento completamente NUEVO")
                print("  [c] Cancelar: Descartar captura")
                
                opcion = input("Elige una opción [s/n/g/c]: ").strip().lower()
                nombre_final = None
                
                if opcion == 's':
                    if modelo is None:
                        print("[!] No hay modelo entrenado. Usa la opción [g].")
                        continue
                    nombre_final = nombre_predicho
                
                elif opcion == 'n':
                    nombres_existentes = [nombre_visible(k) for k in referencias.keys()] #
                    print("\nMovimientos registrados:")
                    for i, n in enumerate(nombres_existentes):
                        print(f"  [{i}] {n}")
                    try:
                        idx_sel = int(input("Introduce el número correcto: "))
                        nombre_final = nombres_existentes[idx_sel]
                    except Exception:
                        print("[!] Selección incorrecta. Secuencia abortada.")
                
                elif opcion == 'g':
                    nuevo_nombre = input("\nIntroduce el nombre de la nueva acción: ").strip().lower()
                    if nuevo_nombre:
                        nombre_final = nuevo_nombre
                else:
                    print("Secuencia descartada.")

                # --- GUARDADO Y RE-ENTRENAMIENTO ---
                if nombre_final:
                    clave_existente = buscar_clave_por_nombre(referencias, nombre_final) #
                    
                    # Guardamos los 30 frames normalizados respetando tu estructura original
                    ruta, total, _ = guardar_referencia( #
                        nombre=nombre_final,
                        secuencia=secuencia_normalizada,
                        clave_existente=clave_existente
                    )
                    print(f"-> Muestra inyectada en referencias.")
                    print(f"   Total de muestras de '{nombre_final}': {total}")
                    
                    # Re-entrenamiento automático instantáneo de la LSTM
                    modelo = entrenar_ia_desde_referencias(modelo)
                    
                buffer_dinamico.clear()
                print("\nListo para el siguiente. Presiona [ESPACIO] para iniciar otra captura.\n")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    hands.close()

if __name__ == "__main__":
    main()