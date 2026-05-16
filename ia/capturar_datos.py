import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- CONFIGURACIÓN ---
DATASET_PATH = "./dataset"
ACCIONES = ["aplaudir", "saludar", "reposo"]
NUM_SECUENCIAS = 30  # Cuántas veces vas a grabar cada movimiento
LARGO_SECUENCIA = 30 # Cuántos frames dura cada movimiento (~1 segundo)

# Asegurar que las carpetas existan
for accion in ACCIONES:
    os.makedirs(os.path.join(DATASET_PATH, accion), exist_ok=True)

# Inicializar MediaPipe Pose de forma ligera
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("=== RECOLECTOR DE MOVIMIENTOS ===")
print("Instrucciones: Elige una acción presionando el número correspondiente en tu teclado:")
for i, accion in enumerate(ACCIONES):
    print(f"Presiona [{i}] para empezar a grabar: '{accion}'")
print("Presiona [q] para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Mostrar preview en espejo para comodidad del usuario
    frame_espejo = cv2.flip(frame, 1)
    cv2.putText(frame_espejo, "SISTEMA LISTO - ESPERANDO ACCION", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Recolector de Datos', frame_espejo)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
    # Si presionas 0, 1 o 2...
    if key in [ord(str(i)) for i in range(len(ACCIONES))]:
        accion_seleccionada = ACCIONES[int(chr(key))]
        
        # Buscar el número de archivo disponible para no sobreescribir datos anteriores
        carpeta_accion = os.path.join(DATASET_PATH, accion_seleccionada)
        archivos_existentes = len(os.listdir(carpeta_accion))
        
        print(f"\nPREPARATE: Grabando muestra #{archivos_existentes} para '{accion_seleccionada}' en 2 segundos...")
        time.sleep(2)
        
        secuencia_datos = []
        contador_frames = 0
        
        while contador_frames < LARGO_SECUENCIA:
            ret, frame = cap.read()
            if not ret: break
            
            # Procesar con MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = pose.process(frame_rgb)
            
            # Extraer los 99 puntos (33 landmarks * X,Y,Z)
            if resultados.pose_landmarks:
                frame_landmarks = []
                for lm in resultados.pose_landmarks.landmark:
                    frame_landmarks.extend([lm.x, lm.y, lm.z])
                
                secuencia_datos.append(frame_landmarks)
                contador_frames += 1
                
                # Feedback en pantalla mientras se graba
                frame_espejo = cv2.flip(frame, 1)
                cv2.putText(frame_espejo, f"GRABANDO: {contador_frames}/{LARGO_SECUENCIA}", (15, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Recolector de Datos', frame_espejo)
                cv2.waitKey(1)
            else:
                # Si el humano se sale del frame, avisamos y no contamos ese frame
                frame_espejo = cv2.flip(frame, 1)
                cv2.putText(frame_espejo, "SITUATE EN FRENTE DE LA CAMARA", (15, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow('Recolector de Datos', frame_espejo)
                cv2.waitKey(1)

        # Guardar la secuencia completa de 30 frames en un archivo numpy binario
        ruta_guardado = os.path.join(carpeta_accion, f"muestra_{archivos_existentes}.npy")
        np.save(ruta_guardado, secuencia_datos)
        print(f"Muestra guardada en: {ruta_guardado}")

cap.release()
cv2.destroyAllWindows()
pose.close()