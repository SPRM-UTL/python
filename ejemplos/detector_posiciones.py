"""
detector_posiciones_v2.py
Versión mejorada con manejo de errores y compatibilidad
"""

import cv2
import time
import math
import os
import sys

# Intenta importar mediapipe con mejor manejo de errores
try:
    import mediapipe as mp
    print("MediaPipe importado correctamente")
except ImportError as e:
    print(f"Error al importar MediaPipe: {e}")
    print("\nPor favor instala MediaPipe con:")
    print("pip install mediapipe==0.10.8")
    sys.exit(1)
except AttributeError as e:
    print(f"Error de atributo en MediaPipe: {e}")
    print("\nEs posible que tengas una versión muy antigua o corrupta.")
    print("Prueba a desinstalar y reinstalar:")
    print("pip uninstall mediapipe -y")
    print("pip install mediapipe==0.10.8")
    sys.exit(1)

# Verificar que la cámara funcione
try:
    cap_test = cv2.VideoCapture(0)
    if not cap_test.isOpened():
        print("Advertencia: No se pudo abrir la camara 0")
    else:
        print("Camara detectada")
        cap_test.release()
except Exception as e:
    print(f"Error al probar la camara: {e}")


class RastreadorCuerpo:
    def __init__(self):
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            self.mp_drawing = mp.solutions.drawing_utils
            print("Rastreador inicializado correctamente")
        except Exception as e:
            print(f"Error al inicializar MediaPipe Pose: {e}")
            raise
        
        self.posicion_anterior = ""
        self.tiempo_ultimo_cambio = 0
        self.tiempo_estabilizacion = 0.3

    def obtener_punto(self, landmarks, indice, imagen_shape):
        """Obtiene coordenadas de un punto específico del cuerpo."""
        if landmarks and indice < len(landmarks.landmark):
            punto = landmarks.landmark[indice]
            h, w = imagen_shape[:2]
            return {
                'x': punto.x,
                'y': punto.y,
                'z': punto.z,
                'visibilidad': punto.visibility,
                'pixel_x': int(punto.x * w),
                'pixel_y': int(punto.y * h)
            }
        return None

    def calcular_angulo(self, p1, p2, p3):
        """Calcula el ángulo formado por tres puntos."""
        if not all([p1, p2, p3]):
            return None
        
        angulo = math.degrees(
            math.atan2(p3['y'] - p2['y'], p3['x'] - p2['x']) -
            math.atan2(p1['y'] - p2['y'], p1['x'] - p2['x'])
        )
        angulo = abs(angulo)
        if angulo > 180:
            angulo = 360 - angulo
        return angulo

    def identificar_posicion(self, landmarks, imagen_shape):
        """Analiza los puntos del cuerpo y determina la posición."""
        if not landmarks:
            return "Sin persona detectada"

        # Obtener puntos clave
        puntos = {}
        for i in range(33):
            puntos[i] = self.obtener_punto(landmarks, i, imagen_shape)

        # Calcular ángulos principales
        angulo_codo_izq = self.calcular_angulo(puntos[11], puntos[13], puntos[15])
        angulo_codo_der = self.calcular_angulo(puntos[12], puntos[14], puntos[16])
        angulo_rodilla_izq = self.calcular_angulo(puntos[23], puntos[25], puntos[27])
        angulo_rodilla_der = self.calcular_angulo(puntos[24], puntos[26], puntos[28])

        # --- DETECCIÓN DE POSICIONES ---
        
        # 1. POSICIÓN T
        if (angulo_codo_izq and angulo_codo_der and
            160 < angulo_codo_izq < 200 and 160 < angulo_codo_der < 200 and
            puntos[15] and puntos[16] and puntos[11] and puntos[12] and
            puntos[15]['y'] < puntos[11]['y'] + 0.05 and
            puntos[16]['y'] < puntos[12]['y'] + 0.05 and
            abs(puntos[15]['x'] - puntos[16]['x']) > 0.4):
            return "POSICION T"

        # 2. BRAZOS ARRIBA
        if (puntos[15] and puntos[16] and puntos[11] and puntos[12] and puntos[0] and
            puntos[15]['y'] < puntos[11]['y'] - 0.1 and
            puntos[16]['y'] < puntos[12]['y'] - 0.1 and
            puntos[15]['y'] < puntos[0]['y'] and
            puntos[16]['y'] < puntos[0]['y']):
            return "BRAZOS ARRIBA"

        # 3. BRAZO DERECHO ARRIBA
        if (puntos[16] and puntos[12] and puntos[0] and
            puntos[16]['y'] < puntos[12]['y'] - 0.1 and
            puntos[16]['y'] < puntos[0]['y'] and
            puntos[15] and puntos[15]['y'] > puntos[11]['y']):
            return "BRAZO DERECHO ARRIBA"

        # 4. BRAZO IZQUIERDO ARRIBA
        if (puntos[15] and puntos[11] and puntos[0] and
            puntos[15]['y'] < puntos[11]['y'] - 0.1 and
            puntos[15]['y'] < puntos[0]['y'] and
            puntos[16] and puntos[16]['y'] > puntos[12]['y']):
            return "BRAZO IZQUIERDO ARRIBA"

        # 5. APLAUSO
        if (puntos[15] and puntos[16] and
            abs(puntos[15]['x'] - puntos[16]['x']) < 0.05 and
            abs(puntos[15]['y'] - puntos[16]['y']) < 0.07):
            return "APLAUSO"

        # 6. MANOS EN LA CINTURA
        if (puntos[15] and puntos[16] and puntos[23] and puntos[24] and
            abs(puntos[15]['x'] - puntos[23]['x']) < 0.1 and
            abs(puntos[16]['x'] - puntos[24]['x']) < 0.1 and
            puntos[15]['y'] > puntos[11]['y'] and
            puntos[16]['y'] > puntos[12]['y']):
            return "MANOS EN LA CINTURA"

        # 7. SENTADILLA
        if (angulo_rodilla_izq and angulo_rodilla_der and
            angulo_rodilla_izq < 120 and angulo_rodilla_der < 120):
            return "SENTADILLA"

        # 8. PIERNA LEVANTADA
        if (puntos[27] and puntos[28] and
            abs(puntos[27]['y'] - puntos[28]['y']) > 0.15):
            if puntos[27]['y'] < puntos[28]['y']:
                return "PIERNA IZQ LEVANTADA"
            else:
                return "PIERNA DER LEVANTADA"

        # 9. BRAZOS CRUZADOS
        if (puntos[15] and puntos[16] and puntos[11] and puntos[12] and
            puntos[15]['x'] > puntos[11]['x'] + 0.1 and
            puntos[16]['x'] < puntos[12]['x'] - 0.1):
            return "BRAZOS CRUZADOS"

        # 10. DE PIE
        if (puntos[15] and puntos[16] and puntos[11] and puntos[12] and
            puntos[15]['y'] > puntos[11]['y'] and
            puntos[16]['y'] > puntos[12]['y'] and
            angulo_codo_izq and 160 < angulo_codo_izq < 200 and
            angulo_codo_der and 160 < angulo_codo_der < 200):
            return "DE PIE"

        return "Movimiento no identificado"


def limpiar_terminal():
    """Limpia la terminal."""
    os.system('cls' if os.name == 'nt' else 'clear')


def main():
    print("=" * 60)
    print("RASTREADOR DE CUERPO COMPLETO")
    print("=" * 60)
    
    # Inicializar rastreador
    try:
        rastreador = RastreadorCuerpo()
    except Exception as e:
        print(f"\nNo se pudo inicializar el rastreador: {e}")
        return
    
    # Inicializar cámara
    print("\nIniciando camara...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la camara 0")
        print("   Prueba a cambiar el número de cámara o verifica conexiones")
        return
    
    print("Todo listo. Presiona 'q' para salir\n")
    
    posicion_mostrada = ""
    frame_count = 0
    
    try:
        while True:
            exito, frame = cap.read()
            if not exito:
                print("Error al leer frame")
                break

            # Espejo
            frame = cv2.flip(frame, 1)
            
            # Procesar
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = rastreador.pose.process(frame_rgb)
            
            # Identificar posición
            posicion = rastreador.identificar_posicion(
                resultados.pose_landmarks, 
                frame.shape
            )
            
            # Dibujar puntos
            if resultados.pose_landmarks:
                rastreador.mp_drawing.draw_landmarks(
                    frame,
                    resultados.pose_landmarks,
                    rastreador.mp_pose.POSE_CONNECTIONS
                )
            
            # Actualizar terminal
            frame_count += 1
            if frame_count % 10 == 0 and posicion != posicion_mostrada:
                limpiar_terminal()
                print("=" * 60)
                print(f"POSICION: {posicion}")
                print("=" * 60)
                print(f"Frame: {frame_count}")
                print("Presiona 'q' para salir")
                posicion_mostrada = posicion
            
            # Mostrar en ventana
            cv2.putText(frame, posicion, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Rastreador de Cuerpo', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nPrograma interrumpido")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Finalizado")


if __name__ == "__main__":
    main()