import cv2
import numpy as np
from ultralytics import YOLO

# Configurações da janela para referência ao dividir seções
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Definições de Direções
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Inicializar o modelo YOLOv8 pré-treinado
model = YOLO("yolov8n.pt")  # Certifique-se de que o modelo YOLOv8 está corretamente configurado

def calculate_center(box):
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy

def determine_direction(center_x, center_y):
    left_section = WINDOW_WIDTH // 3
    right_section = 2 * (WINDOW_WIDTH // 3)
    middle_top = WINDOW_HEIGHT // 2

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

    # Obter as dimensões da janela de captura
    height, width = frame.shape[:2]

    # Redimensionar para se adequar à janela
    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
    frame = cv2.flip(frame, 1)

    # Realizar a detecção usando YOLOv8
    results = model(frame)
    direction = None

    # Processar as detecções
    for result in results:
        for box in result.boxes:
            cls = box.cls[0]  # Classe prevista
            if cls == 67:  # ID 67 é o código COCO para "Mobile Phone"
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas da bounding box
                cx, cy = calculate_center([x1, y1, x2, y2])
                direction = determine_direction(cx, cy)

                # Desenhar bounding box e centro no frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                break

    # Desenha as linhas de referência para dividir a tela
    left_section = WINDOW_WIDTH // 3
    right_section = 2 * (WINDOW_WIDTH // 3)
    middle_top = WINDOW_HEIGHT // 2

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
