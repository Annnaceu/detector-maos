import numpy as np
import cv2
import mediapipe as mp
import joblib
import math

# Carregar o modelo treinado
modelo = joblib.load('modelo_maos_bola.pkl')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.85, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

def distancia(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

simulacao_bola = (320, 240)  # Centro da bola fictícia na tela

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks)

            h, w, _ = frame.shape
            dedo_indicador = hand_landmarks.landmark[8]
            pos_dedo = (int(dedo_indicador.x * w), int(dedo_indicador.y * h))

            # Simular a detecção da bolinha azul
            dist = distancia(pos_dedo, simulacao_bola)
            status_bola = 1 if dist < 50 else 0

            # Extrair características e prever com o modelo
            features = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
            status_predito = modelo.predict([features])[0]

            if status_predito == 1:
                cv2.putText(frame, 'Bola Pega!', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Bola Longe', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Detecção de Mãos e Bola', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
