import numpy as np
import cv2
import mediapipe as mp
import math

# Inicializa os módulos do MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.85, min_tracking_confidence=0.7)

# Simulação da bolinha azul (coordenadas fixas)
simulacao_bola = (320, 240)  # Centro da bola fictícia na tela

cap = cv2.VideoCapture(0)
data = []

def distancia(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    pos_dedos = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks)

            # Extrair as posições dos dedos
            h, w, _ = frame.shape
            dedo_indicador = hand_landmarks.landmark[8]
            pos_dedo = (int(dedo_indicador.x * w), int(dedo_indicador.y * h))
            pos_dedos.append(pos_dedo)

            # Simular a detecção da bolinha azul
            dist = distancia(pos_dedo, simulacao_bola)
            if dist < 50:
                status_bola = 1  # Bolinha perto
            else:
                status_bola = 0  # Bolinha longe

            # Salvar as características (posição dos dedos e status da bola)
            features = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
            data.append((features.tolist(), status_bola))

    cv2.imshow('Coleta de Dados', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Salvar os dados coletados em um arquivo
data = np.array(data, dtype=object)
np.save('dados_maos_bola.npy', data)

cap.release()
cv2.destroyAllWindows()
