# Detector de Lengua de Senas con DETR

Proyecto en Python para capturar imagenes por webcam, generar etiquetas YOLO basicas, entrenar un modelo DETR y ejecutar deteccion en tiempo real.

El proyecto esta dividido en dos partes:

```text
aplicacion/
  app.py              # Deteccion en tiempo real
  boxes.py            # Utilidades de cajas
  config.json         # Clases, colores y camara
  model.py            # Modelo DETR
  pretrained/         # Pesos del modelo

entrenamiento/
  collect_images.py        # Captura fotos por clase
  generate_yolo_labels.py  # Genera labels YOLO automaticos
  data.py                  # Dataset para entrenamiento
  loss.py                  # Loss DETR + Hungarian matcher
  train.py                 # Entrenamiento
  data/
    train/
      images/
      labels/
    test/
      images/
      labels/
  checkpoints/
```

## Requisitos

Instala dependencias con `uv`:

```powershell
uv sync
```

Si usas el entorno virtual directamente:

```powershell
.venv\Scripts\python.exe -m pip install -e .
```

Dependencias principales:

- `torch`
- `torchvision`
- `opencv-python`
- `albumentations`
- `scipy`
- `colorama`

## Configuracion

Las clases, colores y camara se configuran en:

```text
aplicacion/config.json
```

Ejemplo:

```json
{
  "classes": ["hello", "iloveyou", "thankyou"],
  "colors": [
    [41, 171, 202],
    [240, 172, 95],
    [131, 193, 103]
  ],
  "camera": {
    "type": "webcam",
    "source": 0
  }
}
```

Cada clase debe tener un color. El orden importa: ese mismo orden se usa como indice de clase para las etiquetas YOLO.

Para camara wifi puedes usar:

```json
"camera": {
  "type": "wifi",
  "source": "http://192.168.1.4:8080/video"
}
```

## 1. Capturar Fotos

Para capturar imagenes de entrenamiento:

```powershell
.venv\Scripts\python.exe entrenamiento\collect_images.py --num-images 30
```

Por defecto guarda en:

```text
entrenamiento/data/train/images
```

Los nombres quedan con este formato:

```text
hello-xxxxxxxx.jpg
iloveyou-xxxxxxxx.jpg
thankyou-xxxxxxxx.jpg
```

Ese prefijo es importante porque luego se usa para generar la clase automaticamente.

Opciones utiles:

```powershell
.venv\Scripts\python.exe entrenamiento\collect_images.py --num-images 50 --sleep-time 0.5
```

Cambiar salida:

```powershell
.venv\Scripts\python.exe entrenamiento\collect_images.py --output entrenamiento\data\test\images --num-images 10
```

Cambiar camara:

```powershell
.venv\Scripts\python.exe entrenamiento\collect_images.py --camera 1
```

Para detener la captura, presiona `q` en la ventana de OpenCV.

## 2. Generar Etiquetas YOLO Automaticas

Despues de capturar fotos, genera los `.txt` de YOLO:

```powershell
.venv\Scripts\python.exe entrenamiento\generate_yolo_labels.py
```

Esto crea etiquetas en:

```text
entrenamiento/data/train/labels
entrenamiento/data/test/labels
```

Formato generado:

```text
class_id x_center y_center width height
```

Ejemplo:

```text
0 0.500000 0.500000 1.000000 1.000000
```

### Modos de caja

Caja completa:

```powershell
.venv\Scripts\python.exe entrenamiento\generate_yolo_labels.py --mode full
```

Caja centrada:

```powershell
.venv\Scripts\python.exe entrenamiento\generate_yolo_labels.py --mode center --box-size 0.75
```

`--box-size` es el tamano normalizado de la caja. `0.75` significa 75% del ancho y alto de la imagen.

### Separar datos de prueba automaticamente

Si todas tus fotos estan en `train/images`, puedes mover una parte a `test/images`:

```powershell
.venv\Scripts\python.exe entrenamiento\generate_yolo_labels.py --test-ratio 0.2
```

Eso mueve aproximadamente 20% de las imagenes a test antes de generar etiquetas.

### Sobrescribir etiquetas

Si ya existen labels y quieres regenerarlos:

```powershell
.venv\Scripts\python.exe entrenamiento\generate_yolo_labels.py --overwrite
```

## 3. Entrenar

Cuando tengas imagenes y labels en `train` y `test`, ejecuta:

```powershell
.venv\Scripts\python.exe entrenamiento\train.py
```

Opciones utiles:

```powershell
.venv\Scripts\python.exe entrenamiento\train.py --epochs 50 --batch-size 4 --lr 0.00001
```

El entrenamiento usa:

```text
entrenamiento/data/train/images
entrenamiento/data/train/labels
entrenamiento/data/test/images
entrenamiento/data/test/labels
```

Y carga pesos base desde:

```text
aplicacion/pretrained/4426_model.pt
```

Los checkpoints se guardan en:

```text
entrenamiento/checkpoints
```

Ejemplo:

```text
entrenamiento/checkpoints/100_model.pt
```

## 4. Usar el Modelo Entrenado en la Aplicacion

La app de tiempo real carga actualmente:

```text
aplicacion/pretrained/99_model.pt
```

Si entrenaste un modelo nuevo, tienes dos opciones:

1. Copiar el checkpoint nuevo a `aplicacion/pretrained/`.
2. Cambiar `MODEL_WEIGHTS_PATH` en `aplicacion/app.py`.

Ejemplo dentro de `aplicacion/app.py`:

```python
MODEL_WEIGHTS_PATH = os.path.join(
    "entrenamiento",
    "checkpoints",
    "100_model.pt"
)
```

## 5. Ejecutar Deteccion en Tiempo Real

Desde la raiz del proyecto:

```powershell
.venv\Scripts\python.exe main.py
```

Tambien puedes ejecutar directamente:

```powershell
.venv\Scripts\python.exe aplicacion\app.py
```

Para cerrar la ventana de deteccion, presiona `q`.

## Flujo Recomendado

1. Configura clases y colores en `aplicacion/config.json`.
2. Captura fotos:

```powershell
.venv\Scripts\python.exe entrenamiento\collect_images.py --num-images 30
```

3. Genera labels y separa test:

```powershell
.venv\Scripts\python.exe entrenamiento\generate_yolo_labels.py --test-ratio 0.2 --mode center --box-size 0.75
```

4. Entrena:

```powershell
.venv\Scripts\python.exe entrenamiento\train.py --epochs 100
```

5. Apunta la app al checkpoint nuevo.
6. Ejecuta:

```powershell
.venv\Scripts\python.exe main.py
```

## Notas Importantes

- Las etiquetas automaticas no detectan la mano; solo generan una caja aproximada.
- Para mejores resultados, usa fondos variados, diferentes distancias y buena iluminacion.
- Mantener las manos dentro del area de la caja automatica mejora el entrenamiento.
- Si usas `--mode full`, el modelo aprende que el objeto ocupa casi toda la imagen.
- Si usas `--mode center`, procura capturar la sena centrada.
- El nombre de la imagen debe iniciar con una clase valida, por ejemplo `hello-`.
- Si agregas o quitas clases, actualiza tambien `colors` en `aplicacion/config.json`.
- Los pesos `.pt` estan ignorados por Git para evitar subir archivos pesados.

## Problemas Comunes

### No abre la camara

Prueba otro indice:

```powershell
.venv\Scripts\python.exe entrenamiento\collect_images.py --camera 1
```

O revisa `aplicacion/config.json`.

### No se generan labels

Revisa que las imagenes se llamen con el prefijo de clase:

```text
hello-xxxx.jpg
```

Y que `hello` exista en `aplicacion/config.json`.

### El entrenamiento dice que no hay datos suficientes

Con `batch-size 4`, necesitas al menos 4 muestras en train y 4 en test. Puedes bajar el batch:

```powershell
.venv\Scripts\python.exe entrenamiento\train.py --batch-size 1
```

### La app sigue usando el modelo viejo

Cambia `MODEL_WEIGHTS_PATH` en `aplicacion/app.py` para apuntar al checkpoint nuevo.
