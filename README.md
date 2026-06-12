# 🤖 Manordomo - Sistema de Reconocimiento de Gestos para Control Domótico

## Descripción

Manordomo es un sistema de visión por computadora desarrollado en Python que permite identificar movimientos y gestos de la mano mediante Inteligencia Artificial para convertirlos en comandos capaces de controlar dispositivos IoT.

El sistema utiliza técnicas de Machine Learning y procesamiento de video en tiempo real para detectar patrones de movimiento, reconocer referencias visuales y generar acciones que posteriormente pueden ser enviadas a microcontroladores como Arduino o ESP32.

Este proyecto forma parte del Proyecto Integrador de la carrera de Ingeniería en Desarrollo y Gestión de Software de la Universidad Tecnológica de León.

---

## Características

✅ Captura de video en tiempo real

✅ Detección de movimientos mediante visión artificial

✅ Clasificación de gestos utilizando modelos entrenados

✅ Reconocimiento de referencias visuales

✅ Integración con dispositivos IoT

✅ Arquitectura preparada para sistemas domóticos

✅ Entrenamiento y reutilización de modelos de IA

---

## Tecnologías Utilizadas

### Lenguajes

* Python 3

### Librerías principales

* OpenCV
* TensorFlow / Keras
* NumPy
* MediaPipe
* Scikit-Learn

### Hardware compatible

* Arduino
* ESP32
* Módulos Bluetooth
* Dispositivos IoT

---

## Arquitectura General

```text
Cámara Web
      │
      ▼
Captura de Video
      │
      ▼
Procesamiento de Frames
      │
      ▼
Extracción de Características
      │
      ▼
Modelo de IA
      │
      ▼
Clasificación de Gestos
      │
      ▼
Generación de Comando
      │
      ▼
Arduino / ESP32
      │
      ▼
Dispositivo Domótico
```

---

## Estructura del Proyecto

```text
python/
│
├── movimiento/
│   ├── Dataset de movimientos
│   ├── Imágenes de entrenamiento
│   └── Recursos utilizados por el modelo
│
├── referencias/
│   ├── Dataset de referencias visuales
│   └── Recursos auxiliares
│
├── asistente_videos.py
│
├── ia_asistente_delimitado.py
│
├── ia_detector_referencias.h5
│
├── requirements.txt
│
└── README.md
```

---

## Archivos Principales

### asistente_videos.py

Módulo encargado del procesamiento y análisis de video.

Funciones principales:

* Captura de cámara.
* Lectura de frames.
* Procesamiento visual.
* Detección de movimientos.

---

### ia_asistente_delimitado.py

Implementación principal de la lógica de Inteligencia Artificial.

Responsabilidades:

* Carga de modelos entrenados.
* Predicción de movimientos.
* Clasificación de eventos.
* Generación de respuestas del sistema.

---

### ia_detector_referencias.h5

Modelo entrenado de TensorFlow/Keras encargado de reconocer referencias visuales utilizadas por el sistema.

---

## Instalación

### Clonar repositorio

```bash
git clone https://github.com/SPRM-UTL/python.git
cd python
```

### Crear entorno virtual

```bash
python -m venv venv
```

### Activar entorno

Windows:

```bash
venv\Scripts\activate
```

Linux:

```bash
source venv/bin/activate
```

### Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Ejecución

Ejecutar el asistente principal:

```bash
python ia_asistente_delimitado.py
```

o

```bash
python asistente_videos.py
```

---

## Flujo de Funcionamiento

1. La cámara captura video en tiempo real.
2. Se procesan los frames recibidos.
3. El modelo identifica movimientos o referencias.
4. Se clasifica el gesto detectado.
5. Se genera una acción correspondiente.
6. El comando puede enviarse a un dispositivo IoT.
7. El dispositivo ejecuta la acción solicitada.

---

## Aplicaciones

* Casas inteligentes.
* Control de iluminación.
* Automatización industrial.
* Interfaces sin contacto.
* Sistemas de accesibilidad.
* Proyectos educativos de IA e IoT.

---

## Futuras Mejoras

* Aplicación Android nativa.
* Reconocimiento de múltiples manos.
* Mayor catálogo de gestos.
* Integración con MQTT.
* Dashboard Web.
* Comunicación directa con ESP32 mediante WiFi.
* Entrenamiento automático de nuevos gestos.

---

## Autor

José Ángel Cuéllar Gutiérrez

Universidad Tecnológica de León

Ingeniería en Desarrollo y Gestión de Software

Proyecto Integrador - Extracción de Conocimiento de Bases de Datos

---

## Licencia

Proyecto desarrollado con fines académicos y educativos.
