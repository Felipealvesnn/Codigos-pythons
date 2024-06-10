import cv2
import cvzone
import math
from ultralytics import YOLO

# Abrir a webcam
cap = cv2.VideoCapture(0)  # 0 é o índice da webcam

# Verificar se a webcam está aberta corretamente
if not cap.isOpened():
    print("Erro ao abrir a webcam")
    exit()

model = YOLO('yolov8n.pt')

classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

while True:
    ret, frame = cap.read()
    
    # Verificar se o frame foi capturado corretamente
    if not ret:
        print("Erro ao capturar o frame")
        break

    # Redimensionar o frame (opcional)
    frame = cv2.resize(frame, (980, 740))

    # Realizar a detecção de objetos no frame
    results = model(frame)

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)

            # Implementar detecção de queda usando as coordenadas x1, y1, x2, y2
            height = y2 - y1
            width = x2 - x1
            threshold = height - width

            if conf > 80 and class_detect == 'person':
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2)

                if threshold < 0:
                    cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 40], thickness=2, scale=2)

    cv2.imshow('YOLOv8 Fall Detection', frame)

    # Parar o loop se a tecla 't' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('t'):
        break

# Liberar a webcam e fechar todas as janelas
cap.release()
cv2.destroyAllWindows()
