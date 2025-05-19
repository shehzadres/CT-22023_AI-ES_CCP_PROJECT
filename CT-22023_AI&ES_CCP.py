import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define finger states for 30 gestures (thumb, index, middle, ring, pinky)
GESTURES = {
    (0, 0, 0, 0, 0): ("Fist", "Stop or Ready"),
    (1, 1, 1, 1, 1): ("Open Hand", "Hello or Calm"),
    (0, 1, 1, 0, 0): ("Peace", "Victory or Peace"),
    (1, 0, 0, 0, 0): ("Thumbs Up", "Like or Good"),
    (0, 1, 0, 0, 0): ("Pointing", "Point or Attention"),
    (1, 0, 1, 0, 1): ("Rock", "Rock n Roll"),
    (1, 1, 1, 0, 0): ("OK", "Perfect or Fine"),
    (1, 0, 0, 0, 1): ("Call Me", "Phone Gesture"),
    (0, 1, 1, 1, 1): ("Four Fingers", "Number 4"),
    (0, 1, 1, 1, 0): ("Three Fingers", "Number 3"),
    (0, 1, 0, 0, 0): ("One Finger", "Number 1"),
    (1, 1, 0, 0, 0): ("L Sign", "Loser or Left"),
    (0, 1, 0, 1, 0): ("Scissors", "Cut or Snip"),
    (0, 0, 1, 1, 0): ("Middle-Ring", "Number 2 Alt"),
    (1, 1, 0, 0, 1): ("C Sign", "Curve or Hold"),
    (1, 0, 0, 1, 0): ("Y Sign", "Hang Loose"),
    (0, 1, 0, 0, 1): ("Gun", "Bang or Target"),
    (0, 1, 0, 1, 1): ("Fork", "Eat or V Shape"),
    (1, 1, 0, 1, 0): ("Talk to Hand", "Stop Talking"),
    (1, 1, 1, 0, 1): ("Fan", "Wave or Goodbye"),
    (0, 0, 0, 1, 1): ("Bottom Peace", "Low V"),
    (1, 0, 1, 1, 0): ("Crab", "Pinch or Hold"),
    (1, 1, 1, 1, 0): ("High Four", "Number 4 Alt"),
    (0, 0, 1, 0, 0): ("Middle Finger", "Rude or Aggressive"),
    (0, 0, 0, 1, 0): ("Ring Only", "Unknown"),
    (1, 1, 0, 0, 1): ("Split V", "Confuse"),
    (1, 1, 0, 1, 1): ("Fork V", "Unique Gesture"),
    (1, 0, 1, 0, 0): ("Thumb-Middle", "Gesture 29"),
    (0, 0, 0, 0, 1): ("Pinky Only", "Gesture 30"),
}

def finger_is_open(hand_landmarks, tip_id, pip_id):
    return hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y

def recognize_gesture(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    fingers = []

    # Thumb: check x-direction
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[pips[0]].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers: check y-direction
    for i in range(1, 5):
        fingers.append(1 if finger_is_open(hand_landmarks, tips[i], pips[i]) else 0)

    return GESTURES.get(tuple(fingers), ("Unknown", "No matching gesture"))

# Initialize Webcam
cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.75,
                    min_tracking_confidence=0.75) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture_name, gesture_meaning = recognize_gesture(hand_landmarks)

                cv2.putText(frame, f"Gesture: {gesture_name}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Meaning: {gesture_meaning}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Hand Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()
