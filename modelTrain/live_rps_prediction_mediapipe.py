
import cv2
import numpy as np
import mediapipe as mp
import json
from tensorflow.keras.models import load_model

# Load the model and class names
model = load_model("rock_paper_scissors_model.h5")
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    h, w, _ = frame.shape
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Bounding box
            x_min, y_min = w, h
            x_max = y_max = 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            # Add margin
            margin = 30
            x_min, y_min = max(0, x_min - margin), max(0, y_min - margin)
            x_max, y_max = min(w, x_max + margin), min(h, y_max + margin)

            # Crop and predict
            roi = frame[y_min:y_max, x_min:x_max]
            if roi.size > 0:
                roi_resized = cv2.resize(roi, (224, 224))
                roi_normalized = roi_resized.astype("float32") / 255.0
                roi_input = np.expand_dims(roi_normalized, axis=0)

                preds = model.predict(roi_input)
                pred_class = class_names[np.argmax(preds)]

                # Display prediction
                cv2.putText(frame, f"Gesture: {pred_class}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Live Gesture Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
