# Manordomo: Reconocimiento de Gestos con IA e IoT

Este repositorio contiene el backend en Python para el sistema domótico **Manordomo**. Se encarga de capturar el video, procesar las manos usando **MediaPipe** y aplicar modelos de Machine Learning para clasificar gestos y predecir latencia de red.

## Estructura del Proyecto

* `/dataset/`: Contiene los archivos CSV de entrenamiento (landmarks de MediaPipe y logs de telemetría).
* `/src/`: Código fuente principal (extracción de features, scripts de inferencia).
* `/models/`: Modelos pre-entrenados listos para producción (`clasificador_gestos.pkl`, `predictor_latencia.pkl`).
* `/notebooks/`: Libretas de Jupyter utilizadas durante la fase CRISP-DM para EDA y entrenamiento.

## Modelos Incluidos

### 1. Modelo de Clasificación (Gestos)
Utilizado para predecir el gesto actual de la mano a partir de 21 puntos clave (landmarks) 3D extraídos por MediaPipe.
* **Algoritmo:** Random Forest / SVM.
* **Entrada:** Vector de coordenadas (X,Y,Z).
* **Salida:** Clase del gesto (ej. "Puño Cerrado", "Mano Abierta").

### 2. Modelo de Regresión (Optimización IoT)
Utilizado para predecir la latencia (`tiempo_respuesta_ms`) basándose en métricas del sistema para ajustar la transmisión al microcontrolador.
* **Algoritmo:** Regresión Lineal.
* **Entrada:** Nivel de confianza de IA, tamaño de payload, carga de CPU.
* **Salida:** Tiempo en milisegundos.
