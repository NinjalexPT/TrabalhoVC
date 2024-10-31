import cv2
import numpy as np


# Definir os limites para a cor verde
lower_green = np.array([40, 40, 40])  # Limite inferior do verde
upper_green = np.array([80, 255, 255])  # Limite superior do verde

def calculate_center(contour):
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = 0, 0
    return cx, cy



# Inicializa a captura de vídeo com a câmera
cap = cv2.VideoCapture(0)

#Loop para capturar cada frame
while True:
    # Captura o frame
    ret, frame = cap.read()

#Verifica se o frame foi capturado corretamente
    if not ret:
        print("Erro ao capturar o vídeo.")
        break

    # Exibe o frame em uma janela chamada "Câmera"
    cv2.imshow('Câmera', frame)

    # Converter o frame para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Criar uma máscara binária onde o verde é branco e o restante é preto
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Encontrar contornos na máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos e desenhar a bounding box e o centro de massa
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filtra objetos pequenos
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = calculate_center(contour)

            # Desenhar a bounding box e o centro de massa no frame original
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)


    # Exibir a máscara e o frame original com as marcações
    cv2.imshow("Mascara - Objeto Verde", mask)
    cv2.imshow("Camera - Bounding Box e Centro", frame)

    # Sai do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break

#Libera a captura e fecha as janelas
cap.release()
cv2.destroyAllWindows()