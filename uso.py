import numpy as np
import cv2
import mediapipe as mp
import joblib

try:
    modelo = joblib.load('modelo_bolinha.pkl')
except FileNotFoundError:
    print("Modelo não encontrado. Certifique-se de que 'modelo_bolinha.pkl' está no diretório correto.")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.85, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

def contar_dedos(hand_landmarks):
    dedos_levantados = 0
    landmarks = hand_landmarks.landmark

    if landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.THUMB_IP].x:
        dedos_levantados += 1

    dedos = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
             mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    juntas = [mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
              mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.PINKY_PIP]

    for dedo_tip, junta_pip in zip(dedos, juntas):
        if landmarks[dedo_tip].y < landmarks[junta_pip].y:
            dedos_levantados += 1

    return dedos_levantados

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    bolinha_contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bolinha_detectada = False
    for contour in bolinha_contours:
        area = cv2.contourArea(contour)
        if area > 500: 
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            bolinha_detectada = True

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            dedos_levantados = contar_dedos(hand_landmarks)

            features = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
            status_predito = modelo.predict([features])[0]

            cv2.putText(frame, f'Dedos levantados: {dedos_levantados}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, 'Bola Pega!' if bolinha_detectada else 'Bola Longe', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if bolinha_detectada else (0, 0, 255), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks)

    cv2.imshow('Detecção de Dedos e Bolinha', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

