import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("rock_paper_scissors_model.h5")

# Load correct class order
class_names = ['rock', 'paper', 'scissors']  # Replace with actual order if different

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam capture
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    h, w, c = frame.shape
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box from landmarks
            x_min = w
            y_min = h
            x_max = y_max = 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Padding
            margin = 30
            x_min = max(x_min - margin, 0)
            y_min = max(y_min - margin, 0)
            x_max = min(x_max + margin, w)
            y_max = min(y_max + margin, h)

            # Extract and preprocess ROI
            roi = frame[y_min:y_max, x_min:x_max]
            if roi.shape[0] > 0 and roi.shape[1] > 0:
                roi_resized = cv2.resize(roi, (224, 224))
                roi_normalized = roi_resized.astype("float32") / 255.0
                roi_input = np.expand_dims(roi_normalized, axis=0)

                # Prediction
                preds = model.predict(roi_input)
                pred_label = class_names[np.argmax(preds)]

                # Draw prediction
                cv2.putText(frame, f'Gesture: {pred_label}', (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Live Gesture Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
