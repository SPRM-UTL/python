import os
import sys
import time
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Silenciar logs innecesarios de TensorFlow para mantener limpia la consola
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Reutilizar tus utilidades nativas del repositorio
from movimiento.utilidades_pose import (
    cargar_referencias,
    frame_a_vector_completo,
    guardar_referencia,
    nombre_visible,
    buscar_clave_por_nombre
)

# Parche de compatibilidad obligatorio para tu entorno Protobuf/TensorFlow
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
    """ Detecta el tamaño del vector (pose + manos + extras) inspeccionando una muestra """
    for clave, muestras in referencias.items():
        if muestras and len(muestras[0]) > 0:
            return np.array(muestras[0]).shape[-1]
    return 105 

def construir_o_adaptar_modelo(num_clases, tam_input, modelo_anterior=None):
    """ Crea o muta la red neuronal adaptando la salida al número de carpetas en referencias """
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
        # Transferencia de conocimiento (Fine-Tuning)
        for i in range(len(model.layers) - 1):
            try:
                model.layers[i].set_weights(modelo_anterior.layers[i].get_weights())
            except Exception:
                pass
    return model

def entrenar_ia_desde_referencias(modelo_actual):
    """ Lee todas las subcarpetas de `./referencias/` y entrena la red LSTM """
    referencias = cargar_referencias() #
    if not referencias:
        print("[IA] No hay movimientos en la carpeta 'referencias' para entrenar.")
        return modelo_actual

    tam_input = descubrir_dimension_vector(referencias)
    claves_ordenadas = sorted(list(referencias.keys()))
    label_map = {clave: num for num, clave in enumerate(claves_ordenadas)}
    
    secuencias, etiquetas = [], []
    print(f"\n[IA] Preparando re-entrenamiento con {len(claves_ordenadas)} clases...")
    
    for clave, muestras in referencias.items():
        for muestra in muestras:
            arr_muestra = np.array(muestra, dtype=np.float32)
            if arr_muestra.shape == (LARGO_SECUENCIA, tam_input):
                secuencias.append(arr_muestra)
                etiquetas.append(label_map[clave])
                
    if len(secuencias) < 2:
        print("[IA] Datos insuficientes para entrenar (registra más movimientos).")
        return modelo_actual

    X = np.array(secuencias)
    y = to_categorical(etiquetas, num_classes=len(claves_ordenadas)).astype(int)
    
    if modelo_actual is None or modelo_actual.layers[-1].output_shape[-1] != len(claves_ordenadas):
        print("[IA] Rediseñando salida de la red...")
        modelo_actual = construir_o_adaptar_modelo(len(claves_ordenadas), tam_input, modelo_anterior=modelo_actual)

    print("[IA] Ajustando sinapsis de memoria temporal...")
    modelo_actual.fit(X, y, epochs=35, batch_size=4, verbose=1)
    modelo_actual.save(MODEL_NAME)
    print("[IA] ¡Cerebro actualizado y guardado!\n")
    return modelo_actual

def main():
    referencias = cargar_referencias() #
    
    modelo = None
    if os.path.exists(MODEL_NAME):
        print(f"Cargando modelo neuronal existente ({MODEL_NAME})...")
        modelo = load_model(MODEL_NAME)
        # Sincronizar por si añadiste carpetas manualmente desde el detector clásico
        if referencias and modelo.layers[-1].output_shape[-1] != len(referencias):
            modelo = entrenar_ia_desde_referencias(modelo)
    else:
        if referencias:
            print("Inicializando primer entrenamiento de IA con tus referencias...")
            modelo = entrenar_ia_desde_referencias(modelo)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) #

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    
    # model_complexity=0 garantiza procesamiento ultra fluido sin lag en la cámara
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) #
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) #

    buffer_movimiento = deque(maxlen=LARGO_SECUENCIA) #
    
    print("\n" + "="*65)
    print("  ASISTENTE DE APRENDIZAJE ACTIVO INTERACTIVO POR IA")
    print("="*65)
    print(" Instrucciones:")
    print("  1. Haz el movimiento de forma continua frente a la cámara.")
    print("  2. Presiona [ESPACIO] justo al terminarlo para analizarlo.")
    print("  3. El sistema te dirá qué cree que es y tú elegirás el destino.")
    print("  4. Presiona [q] para salir.")
    print("-"*65 + "\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #
        resultados_pose = pose.process(frame_rgb) #
        resultados_manos = hands.process(frame_rgb) #
        vista = cv2.flip(frame, 1) #

        # Estado en pantalla (La IA está en reposo absoluto consumiendo 0% de CPU aquí)
        texto_ui = "Camara Activa - Historial: "
        if resultados_pose.pose_landmarks:
            vector = frame_a_vector_completo(resultados_pose.pose_landmarks, resultados_manos) #
            buffer_movimiento.append(vector)
            texto_ui += f"{len(buffer_movimiento)}/{LARGO_SECUENCIA} frames listos"
            color_ui = (0, 255, 0) if len(buffer_movimiento) == LARGO_SECUENCIA else (0, 255, 255)
        else:
            texto_ui = "SITUATE EN EL ENCUADRE"
            color_ui = (0, 0, 255)

        cv2.putText(vista, texto_ui, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_ui, 2) #
        cv2.putText(vista, "[Espacio] Analizar / Enseñar  -  [Q] Salir", (15, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Asistente Interactivo", vista)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
        if key == ord(' '):
            if len(buffer_movimiento) < LARGO_SECUENCIA: #
                print(f"[!] Buffer incompleto ({len(buffer_movimiento)}/30). Muévete un instante antes.")
                continue
            
            # --- FASE DE ANÁLISIS ---
            print("\n" + "*"*50)
            print(" [PROCESANDO SECUENCIA CAPTURADA EN MEMORIA]")
            print("*"*50)
            
            referencias = cargar_referencias()
            claves_ordenadas = sorted(list(referencias.keys()))
            nombre_predicho = "Ninguno (IA no entrenada)"
            confianza = 0.0

            # Despertar a la IA únicamente para resolver esta ráfaga de frames
            if modelo is not None and referencias:
                entrada_ia = np.expand_dims(np.array(buffer_movimiento, dtype=np.float32), axis=0)
                prediccion = modelo.predict(entrada_ia, verbose=0)[0]
                idx = np.argmax(prediccion)
                confianza = prediccion[idx]
                nombre_predicho = nombre_visible(claves_ordenadas[idx]) #

            print(f"\n[IA Veredicto]: Detectado candidato -> '{nombre_predicho.upper()}' con {confianza*100:.1f}% de seguridad.")
            print("\n¿Qué deseas hacer con esta secuencia?")
            print(f" [s] Confirmar: Sí, efectivamente fue '{nombre_predicho}'")
            print(" [n] Corregir: Fue otro movimiento de la lista existente")
            print(" [g] Crear: Es un movimiento totalmente NUEVO")
            print(" [c] Cancelar: Descartar esta captura")
            
            opcion = input("Selecciona una opción [s/n/g/c]: ").strip().lower()
            
            nombre_final = None
            
            if opcion == 's':
                if modelo is None:
                    print("[!] No puedes confirmar un movimiento si la IA no está entrenada. Elige [g].")
                    continue
                nombre_final = nombre_predicho
            
            elif opcion == 'n':
                nombres_existentes = [nombre_visible(k) for k in referencias.keys()] #
                print("\nMovimientos disponibles en 'referencias':")
                for i, n in enumerate(nombres_existentes):
                    print(f"  [{i}] {n}")
                try:
                    idx_sel = int(input("Introduce el número correcto: "))
                    nombre_final = nombres_existentes[idx_sel]
                except Exception:
                    print("[!] Selección inválida. Captura abortada.")
            
            elif opcion == 'g':
                nuevo_nombre = input("\nIntroduce el nombre para el nuevo movimiento: ").strip().lower()
                if nuevo_nombre:
                    nombre_final = nuevo_nombre
            
            else:
                print("Captura descartada.")
            
            # --- FASE DE APRENDIZAJE / GUARDADO ---
            if nombre_final:
                # Buscar si el nombre ya tiene una clave UUID asignada en disco
                clave_existente = buscar_clave_por_nombre(referencias, nombre_final) #
                
                # Guardar físicamente usando tu misma función nativa del proyecto
                ruta, total, _ = guardar_referencia( #
                    nombre=nombre_final,
                    secuencia=list(buffer_movimiento),
                    clave_existente=clave_existente
                )
                print(f"-> Muestra guardada con éxito en referencias.")
                print(f"   Muestras totales de '{nombre_final}': {total}")
                
                # Re-entrenamiento automático instantáneo de la Red LSTM
                modelo = entrenar_ia_desde_referencias(modelo)
                
            buffer_movimiento.clear() #
            print("\nSistema listo. Regresa frente a la cámara.\n")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    hands.close()

if __name__ == "__main__":
    main()