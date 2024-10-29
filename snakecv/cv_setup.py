import cv2
import numpy as np

# Inicializa a captura de vídeo com a câmera padrão (geralmente a câmera integrada)
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

    # Sai do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break

#Libera a captura e fecha as janelas
cap.release()
cv2.destroyAllWindows()