from ultralytics import YOLO
import cv2

# Modelo preentrenado
model = YOLO("yolov8n.pt")

# Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # Detectar objetos
    results = model(frame)

    # Dibujar resultados
    annotated_frame = results[0].plot()

    # Mostrar
    cv2.imshow("Object Detection", annotated_frame)

    # Salir con Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()