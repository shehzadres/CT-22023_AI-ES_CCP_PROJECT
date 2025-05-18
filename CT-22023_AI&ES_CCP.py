import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class GestureDetector:
    def __init__(self):
        self.tip_ids = [4, 8, 12, 16, 20]  # Thumb to Pinky tips
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )

    def get_finger_state(self, landmarks):
        fingers = []
        
        # Thumb (compare x-coordinate for right/left hand)
        if landmarks.landmark[self.tip_ids[0]].x < landmarks.landmark[self.tip_ids[0] - 1].x:
            fingers.append(1)  # Thumb open
        else:
            fingers.append(0)  # Thumb closed
        
        # Other fingers (compare y-coordinate)
        for id in range(1, 5):
            if landmarks.landmark[self.tip_ids[id]].y < landmarks.landmark[self.tip_ids[id] - 2].y:
                fingers.append(1)  # Finger open
            else:
                fingers.append(0)  # Finger closed
        return fingers

    def recognize_gesture(self, fingers):
        # Single-hand gestures
        if fingers == [0, 0, 0, 0, 0]:
            return "FIST", "✊ Stop/Ready"
        elif fingers == [1, 1, 1, 1, 1]:
            return "OPEN HAND", "✋ Hello"
        elif fingers == [0, 1, 1, 0, 0]:
            return "PEACE", "✌️ Peace"
        elif fingers == [0, 1, 0, 0, 0]:
            return "POINT", "👆 Select"
        elif fingers == [1, 0, 0, 0, 0]:
            return "THUMBS UP", "👍 Good"
        elif fingers == [0, 1, 0, 0, 1]:
            return "ROCK", "🤘 Rock on!"
        elif fingers == [1, 0, 0, 0, 1]:
            return "SHAKA", "🤙 Hang loose"
        elif fingers == [0, 0, 0, 0, 1]:
            return "PINKY", "Promise"
        else:
            return "UNKNOWN", ""

    def draw_debug_info(self, frame, landmarks, fingers, gesture):
        # Draw hand landmarks
        mp_drawing.draw_landmarks(
            frame, landmarks, mp_hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style()
        )

        # Display finger state (1=open, 0=closed)
        cv2.putText(frame, f'Fingers: {fingers}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display detected gesture
        if gesture[0] != "UNKNOWN":
            cv2.putText(frame, f'{gesture[0]} {gesture[1]}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Try clearer gestures", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

# Main loop
cap = cv2.VideoCapture(0)
detector = GestureDetector()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            fingers = detector.get_finger_state(hand_landmarks)
            gesture = detector.recognize_gesture(fingers)
            frame = detector.draw_debug_info(frame, hand_landmarks, fingers, gesture)

    cv2.imshow('Gesture Control', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()