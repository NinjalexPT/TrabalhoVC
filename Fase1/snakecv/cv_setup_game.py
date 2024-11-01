import cv2
import numpy as np


# Definições de Direções
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Definindo os limites da cor verde em HSV
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

def calculate_center(contour):
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = 0, 0
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

    frame = cv2.flip(frame, 1)
    # Converter o frame para HSV e aplicar máscara para verde
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    direction = None

    # Desenha as linhas de referência para dividir a tela
    left_section = width // 3
    right_section = 2 * (width // 3)
    middle_top = height // 2

    # Linha vertical esquerda (separando área da esquerda)
    cv2.line(frame, (left_section, 0), (left_section, height), (255, 0, 0), 2)
    # Linha vertical direita (separando área da direita)
    cv2.line(frame, (right_section, 0), (right_section, height), (255, 0, 0), 2)
    # Linha horizontal ao meio (separando cima e baixo na área central)
    cv2.line(frame, (left_section, middle_top), (right_section, middle_top), (255, 0, 0))

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cx, cy = calculate_center(largest_contour)
        direction = determine_direction(cx, cy,width,height)

        # Desenhar bounding box e centro no frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    return direction, frame
