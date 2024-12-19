import cv2
from ultralytics import YOLO  # YOLOv8

# Carregar o modelo YOLOv8n treinado
model = YOLO("yolov8n.pt")  # Certifique-se de que o modelo está disponível

# Configuração global para rastreamento
tracker = None
tracking_active = False
bbox = None

# Dimensões dinâmicas da janela
WINDOW_WIDTH = 0
WINDOW_HEIGHT = 0

# Direções
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

def determine_direction(center_x, center_y):
    """Determina a direção com base no centro do bounding box."""
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


def detect_largest_mobile(frame):
    """
    Detecta telemóveis na câmera usando YOLOv8 e seleciona o maior bounding box.
    Retorna o maior bounding box e o frame anotado.
    """
    global WINDOW_WIDTH, WINDOW_HEIGHT

    # Realizar a inferência
    results = model(frame, conf=0.5)  # Confiança mínima de 50%
    annotated_frame = frame.copy()

    largest_bbox = None
    max_area = 0

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas do bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Filtrar apenas telemóveis (classe "cell phone")
            if model.names[cls] == "cell phone":
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    largest_bbox = (x1, y1, x2 - x1, y2 - y1)

                # Desenhar o bounding box para visualização
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Exibir o ponto central
                cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)

                # Exibir a classe e confiança
                label = f"{model.names[cls]}: {confidence:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return largest_bbox, annotated_frame


def get_direction_from_camera(cap):
    """
    Obtém a direção para o jogo com base no tracking de movimento.
    - Exibe a janela apenas para seleção.
    - Filtra para o maior telemóvel detectado.
    """
    global tracking_active, tracker, bbox, WINDOW_WIDTH, WINDOW_HEIGHT

    # Ler frame da câmera
    ret, frame = cap.read()
    if not ret:
        return None, frame

    frame = cv2.flip(frame, 1)  # Espelhar a câmera para melhor jogabilidade
    height, width, _ = frame.shape
    WINDOW_WIDTH, WINDOW_HEIGHT = width, height

    # Se não há tracking ativo, usa YOLO para detecção
    if not tracking_active:
        bbox, annotated_frame = detect_largest_mobile(frame)

        # Mostrar instruções para o jogador
        cv2.putText(
            annotated_frame,
            "Press 'T' to track detected object",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Camera - YOLOv8 Tracking", annotated_frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            return None, frame  # Encerra o programa
        elif key == ord("t") and bbox is not None:
            # Inicia o rastreamento com o bounding box detectado
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, bbox)
            tracking_active = True
            cv2.destroyWindow("Camera - YOLOv8 Tracking")  # Fecha a janela após seleção
            return None, frame

    # Tracking ativo
    if tracker is not None and tracking_active:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            center_x, center_y = x + w // 2, y + h // 2

            # Desenhar o bounding box e o centro
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # Determina a direção e retorna
            direction = determine_direction(center_x, center_y)
            return direction, frame
        else:
            # Tracking falhou
            tracking_active = False
            tracker = None
            return None, frame
    else:
        return None, frame