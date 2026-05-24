import os
import json
import time
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from aplicacion.model import DETR
from aplicacion.boxes import rescale_bboxes

CONFIG_PATH = os.path.join("aplicacion", "config.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

CLASSES = config["classes"]
COLORS = config["colors"]

transforms = A.Compose(
    [   
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
)

print("Cargando modelo DETR...")
model = DETR(num_classes=3)
model.eval()
MODEL_WEIGHTS_PATH = os.path.join("aplicacion", "pretrained", "4426_model.pt")
model.load_pretrained(MODEL_WEIGHTS_PATH)

print("Iniciando captura de video en tiempo real...")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el cuadro de la cámara.")
        break

    transformed = transforms(image=frame)
    input_tensor = torch.unsqueeze(transformed['image'], dim=0)

    with torch.no_grad():
        result = model(input_tensor)

    probabilities = result['pred_logits'].softmax(-1)[:, :, :-1] 
    max_probs, max_classes = probabilities.max(-1)
    
    keep_mask = max_probs > 0.8
    batch_indices, query_indices = torch.where(keep_mask) 

    h, w, _ = frame.shape
    bboxes = rescale_bboxes(result['pred_boxes'][batch_indices, query_indices, :], (w, h))
    classes = max_classes[batch_indices, query_indices]
    probas = max_probs[batch_indices, query_indices]

    for bclass, bprob, bbox in zip(classes, probas, bboxes): 
        idx = bclass.item()
        prob_val = bprob.item() 
        x1, y1, x2, y2 = bbox.tolist()
        
        color = COLORS[idx]
        bgr_color = (color[2], color[1], color[0])

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), bgr_color, 4)
        
        frame_text = f"{CLASSES[idx]} - {round(prob_val, 2)}"
        cv2.rectangle(frame, (int(x1), int(y1) - 35), (int(x1) + 300, int(y1)), bgr_color, -1)
        cv2.putText(frame, frame_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Detección de Lengua de Señas', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        print("Deteniendo detección...")
        break

cap.release() 
cv2.destroyAllWindows()
