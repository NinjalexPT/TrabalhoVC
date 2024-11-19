import cv2
import numpy as np
from ultralytics import YOLO

# Definições de Direções
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Inicializar o modelo YOLOv8 pré-treinado
model = YOLO("yolov8n.pt")  # Certifique-se de que o modelo YOLOv8 está corretamente configurado

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

def calculate_center(box):
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy


def determine_direction(center_x, center_y,width,height):
    left_section = width // 3
    right_section = 2 * (width // 3)
    middle_top = height // 2

    if center_x < left_section:
        return LEFT
    elif center_x > right_section:
        return RIGHT
    elif left_section <= center_x <= right_section:
        if center_y < middle_top:
            return UP
        else:
            return DOWN


def get_direction_from_camera(cap):
    ret, frame = cap.read()
    if not ret:
        return None, frame

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Redimensionar para se adequar à janela
    frame = cv2.resize(frame, (width, height))
    frame = cv2.flip(frame, 1)

    # Realizar a detecção usando YOLOv8
    results = model(frame)
    direction = None

    # Processar as detecções
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # Classe prevista
            confidence = box.conf[0]  # Confiança da detecção
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas da bounding box

            # Apenas lidar com telemóveis
            if cls == 67:  # ID 67 é o código COCO para "cell phone"
                cx, cy = calculate_center([x1, y1, x2, y2])
                direction = determine_direction(cx, cy, width, height)

                # Desenhar bounding box e centro no frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # Escrever o nome da classe e a confiança acima da bounding box
                label = f"{COCO_CLASSES[cls]}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Desenha as linhas de referência para dividir a tela
    left_section = width // 3
    right_section = 2 * (width // 3)
    middle_top = height // 2

    # Linha vertical esquerda (separando área da esquerda)
    cv2.line(frame, (left_section, 0), (left_section, height), (255, 0, 0), 2)
    # Linha vertical direita (separando área da direita)
    cv2.line(frame, (right_section, 0), (right_section, height), (255, 0, 0), 2)
    # Linha horizontal ao meio (separando cima e baixo na área central)
    cv2.line(frame, (left_section, middle_top), (right_section, middle_top), (255, 0, 0), 2)

    return direction, frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        direction, frame = get_direction_from_camera(cap)
        if frame is not None:
            cv2.imshow("Detecção de Telemóvel", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
