import cv2
import numpy as np
from ultralytics import YOLO

# Carrega o modelo YOLOv8
model = YOLO('yolov8n.pt')  # Você pode escolher o modelo que desejar

def is_running(velocities, threshold=5.0):
    """
    Função para determinar se a velocidade indica que a pessoa está correndo.
    `velocities` é uma lista de velocidades calculadas.
    `threshold` é a velocidade acima da qual consideramos que a pessoa está correndo.
    """
    for velocity in velocities:
        if velocity > threshold:
            return True
    return False

# Abre a webcam
cap = cv2.VideoCapture(0)

# Dicionário para armazenar as posições das pessoas entre frames
person_positions = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Faz a detecção usando YOLOv8
    results = model(frame)
    
    # Acessa os boxes dos resultados
    detections = results[0].boxes

    velocities = []
    for result in detections:
        x_min, y_min, x_max, y_max = result.xyxy[0]
        confidence = result.conf[0]
        cls = result.cls[0]

        if int(cls) == 0:  # Classe 0 é 'pessoa' em COCO
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            # Calcula a velocidade
            person_id = (int(x_min), int(y_min), int(x_max), int(y_max))
            if person_id in person_positions:
                old_x, old_y = person_positions[person_id]
                velocity = np.sqrt((center_x - old_x) ** 2 + (center_y - old_y) ** 2)
                velocities.append(velocity)
            
            # Atualiza a posição
            person_positions[person_id] = (center_x, center_y)

            # Desenha o retângulo e a label no frame
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
            cv2.putText(frame, "Person", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    if is_running(velocities):
        cv2.putText(frame, "Person Running Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Mostra o frame processado
    cv2.imshow('YOLOv8 Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a webcam e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()
