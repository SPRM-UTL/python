import os
import sys
import time
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Silenciar logs aparatosos de TensorFlow para mantener la consola limpia
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

# Parche de compatibilidad obligatorio para Protobuf / TensorFlow
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
    """ Inspecciona una muestra física para saber el tamaño del vector (pose + manos + extras) """
    for clave, muestras in referencias.items():
        if muestras and len(muestras[0]) > 0:
            return np.array(muestras[0]).shape[-1]
    return 105 

def construir_o_adaptar_modelo(num_clases, tam_input, modelo_anterior=None):
    """ Modifica la red LSTM mutando la última capa según la cantidad de referencias reales """
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
        # Transferencia de conocimiento (Fine-Tuning) para no olvidar lo ya aprendido
        for i in range(len(model.layers) - 1):
            try:
                model.layers[i].set_weights(modelo_anterior.layers[i].get_weights())
            except Exception:
                pass
    return model

def entrenar_ia_desde_referencias(modelo_actual):
    """ Carga todas las carpetas de `./referencias/` y entrena el modelo de Keras """
    referencias = cargar_referencias() #
    if not referencias:
        print("[IA] No hay carpetas dentro de 'referencias' para entrenar.")
        return modelo_actual

    tam_input = descubrir_dimension_vector(referencias)
    claves_ordenadas = sorted(list(referencias.keys()))
    label_map = {clave: num for num, label in enumerate(claves_ordenadas)}
    
    secuencias, etiquetas = [], []
    print(f"\n[IA] Entrenando la Red Neuronal con {len(claves_ordenadas)} clases de movimientos...")
    
    for clave, muestras in referencias.items():
        for muestra in muestras:
            arr_muestra = np.array(muestra, dtype=np.float32)
            if arr_muestra.shape == (LARGO_SECUENCIA, tam_input):
                secuencias.append(arr_muestra)
                etiquetas.append(label_map[clave])
                
    if len(secuencias) < 2:
        print("[IA] Datos insuficientes globales en referencias para entrenar estandares LSTM.")
        return modelo_actual

    X = np.array(secuencias)
    y = to_categorical(etiquetas, num_classes=len(claves_ordenadas)).astype(int)
    
    # Si cambió la cantidad de carpetas, expandimos estructuralmente la salida de la red
    if modelo_actual is None or modelo_actual.layers[-1].output_shape[-1] != len(claves_ordenadas):
        modelo_actual = construir_o_adaptar_modelo(len(claves_ordenadas), tam_input, modelo_anterior=modelo_actual)

    modelo_actual.fit(X, y, epochs=35, batch_size=4, verbose=1)
    modelo_actual.save(MODEL_NAME)
    print("[IA] ¡Cerebro LSTM sincronizado con éxito!\n")
    return modelo_actual

def main():
    referencias = cargar_referencias() #
    
    modelo = None
    if os.path.exists(MODEL_NAME):
        print(f"Cargando modelo neuronal: {MODEL_NAME}")
        modelo = load_model(MODEL_NAME)
        # Sincronización automática inicial por si hay cambios en el disco
        if referencias and modelo.layers[-1].output_shape[-1] != len(referencias):
            modelo = entrenar_ia_desde_referencias(modelo)
    else:
        if referencias:
            print("No se encontró el archivo .h5. Creando primer entrenamiento...")
            modelo = entrenar_ia_desde_referencias(modelo)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) #

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    
    # model_complexity=0 optimiza drásticamente el uso de CPU, eliminando cualquier tipo de colapso gráfico
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) #
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) #

    buffer_movimiento = deque(maxlen=LARGO_SECUENCIA) #
    
    print("\n" + "="*70)
    print("  ASISTENTE INTERACTIVO POR IA: CAPTURA INMEDIATA AL CLIC DE ESPACIO")
    print("="*70)
    print(" Flujo de trabajo:")
    print("  1. Haz tu movimiento libremente (la cámara va fluida, IA en reposo).")
    print("  2. Presiona [ESPACIO] EXACTAMENTE al terminar el movimiento.")
    print("  3. La IA analizará los últimos 30 frames y te preguntará en consola.")
    print("  4. Presiona [q] para salir.")
    print("-"*70 + "\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #
        resultados_pose = pose.process(frame_rgb) #
        resultados_manos = hands.process(frame_rgb) #
        vista = cv2.flip(frame, 1) #

        # Llenado dinámico continuo del buffer en segundo plano (0% uso de IA aquí)
        if resultados_pose.pose_landmarks:
            vector = frame_a_vector_completo(resultados_pose.pose_landmarks, resultados_manos) #
            buffer_movimiento.append(vector)
            
            estado_texto = f"Grabando flujo... Buffer: {len(buffer_movimiento)}/30"
            color_texto = (0, 255, 0) if len(buffer_movimiento) == LARGO_SECUENCIA else (0, 255, 255)
        else:
            estado_texto = "COLOCATE FRENTE A LA CAMARA"
            color_texto = (0, 0, 255)

        cv2.putText(vista, estado_texto, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_texto, 2) #
        cv2.putText(vista, "[Espacio] Congelar e Investigar Gesto  -  [Q] Salir", (15, 455), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Asistente por IA (Ráfaga Inmediata)", vista)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
        if key == ord(' '):
            if len(buffer_movimiento) < LARGO_SECUENCIA: #
                print(f"[!] Espera a que el buffer se llene ({len(buffer_movimiento)}/30).")
                continue
            
            # --- CONGELACIÓN EN EL ACTO Y ANÁLISIS ---
            print("\n" + "*"*60)
            print(" ¡RAFAGA CONGELADA! PROCESANDO EN LA RED NEURONAL...")
            print("*"*60)
            
            referencias = cargar_referencias()
            claves_ordenadas = sorted(list(referencias.keys()))
            nombre_predicho = "Ninguno (Sin modelo aún)"
            confianza = 0.0

            # Despertamos a la IA únicamente para procesar los 30 frames congelados
            if modelo is not None and referencias:
                # Convertimos el buffer exacto a un formato compatible con Keras (1, 30, dimension)
                entrada_ia = np.expand_dims(np.array(buffer_movimiento, dtype=np.float32), axis=0)
                prediccion = modelo.predict(entrada_ia, verbose=0)[0]
                idx = np.argmax(prediccion)
                confianza = prediccion[idx]
                nombre_predicho = nombre_visible(claves_ordenadas[idx]) #

            print(f"\n[Resultado IA]: Creo que es '{nombre_predicho.upper()}' con un {confianza*100:.1f}% de certeza.")
            print("\n¿Qué dictaminas para esta ráfaga de datos?")
            print(f"  [s] Sí, el veredicto es correcto (Añadir muestra a '{nombre_predicho}')")
            print("  [n] No, fue otro movimiento de la lista existente")
            print("  [g] Es un movimiento completamente NUEVO (Crear etiqueta)")
            print("  [c] Cancelar / Falso positivo (Descartar secuencia)")
            
            opcion = input("Elige una opción [s/n/g/c]: ").strip().lower()
            nombre_final = None
            
            if opcion == 's':
                if modelo is None:
                    print("[!] No hay modelo entrenado aún. Selecciona la opción [g].")
                    continue
                nombre_final = nombre_predicho
            
            elif opcion == 'n':
                nombres_existentes = [nombre_visible(k) for k in referencias.keys()] #
                print("\nMovimientos disponibles:")
                for i, n in enumerate(nombres_existentes):
                    print(f"  [{i}] {n}")
                try:
                    idx_sel = int(input("Introduce el número de la acción real: "))
                    nombre_final = nombres_existentes[idx_sel]
                except Exception:
                    print("[!] Opción inválida. Secuencia abortada.")
            
            elif opcion == 'g':
                nuevo_nombre = input("\nEscribe el nombre del nuevo movimiento: ").strip().lower()
                if nuevo_nombre:
                    nombre_final = nuevo_nombre
            else:
                print("Muestra descartada.")

            # --- FASE DE APRENDIZAJE ---
            if nombre_final:
                # Comprobar si el nombre ya cuenta con un UUID existente en el disco
                clave_existente = buscar_clave_por_nombre(referencias, nombre_final) #
                
                # Guardar el archivo .npy usando las utilidades oficiales de tu ERP/módulo
                ruta, total, _ = guardar_referencia( #
                    nombre=nombre_final,
                    secuencia=list(buffer_movimiento),
                    clave_existente=clave_existente
                )
                print(f"-> Guardado en referencias con éxito.")
                print(f"   Muestras totales de '{nombre_final}': {total}")
                
                # Re-entrenamiento evolutivo adaptando los pesos de inmediato
                modelo = entrenar_ia_desde_referencias(modelo)
                
            buffer_movimiento.clear() #
            print("\nAsistente listo de nuevo. Regresa al encuadre.\n")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    hands.close()

if __name__ == "__main__":
    main()