import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Detect open/closed finger
def finger_is_open(lm, tip, dip):
    return lm[tip].y < lm[dip].y

def recognize_letter(lm):
    tips = [4, 8, 12, 16, 20]
    dips = [3, 7, 11, 15, 19]
    fingers = [0] * 5

    fingers[0] = 1 if lm[4].x < lm[3].x else 0
    for i in range(1, 5):
        fingers[i] = 1 if finger_is_open(lm, tips[i], dips[i]) else 0

    pattern = tuple(fingers)
    z_diff_index = abs(lm[8].z - lm[0].z)
    z_diff_middle = abs(lm[12].z - lm[0].z)
    dist_thumb_index = abs(lm[4].x - lm[8].x)

    if pattern == (0, 0, 0, 0, 0): return "A"
    elif pattern == (0, 1, 1, 1, 1): return "B"
    elif pattern == (0, 1, 1, 0, 0) and z_diff_index < 0.03: return "C"
    elif pattern == (0, 1, 1, 0, 0) and z_diff_index > 0.03: return "D"
    elif pattern == (0, 1, 1, 1, 0): return "E"
    elif pattern == (0, 1, 0, 0, 1): return "F"
    elif pattern == (1, 0, 0, 0, 0): return "G"
    elif pattern == (1, 1, 0, 0, 0): return "H"
    elif pattern == (0, 1, 0, 0, 0): return "I"
    elif pattern == (0, 1, 0, 0, 0) and lm[4].y < lm[3].y: return "J"
    elif pattern == (1, 0, 1, 0, 1): return "K"
    elif pattern == (1, 0, 0, 0, 1): return "L"
    elif pattern == (1, 1, 1, 1, 1): return "M"
    elif pattern == (1, 1, 1, 1, 0): return "N"
    elif pattern == (0, 1, 1, 0, 1): return "O"
    elif pattern == (0, 1, 1, 1, 1): return "P"
    elif pattern == (0, 1, 1, 1, 0) and lm[20].x > lm[16].x: return "Q"
    elif pattern == (0, 1, 0, 1, 1): return "R"
    elif pattern == (0, 0, 0, 0, 1): return "S"
    elif pattern == (0, 0, 0, 0, 1) and lm[4].x > lm[3].x: return "T"
    elif pattern == (0, 1, 0, 0, 1): return "U"
    elif pattern == (0, 1, 0, 0, 1) and dist_thumb_index < 0.05: return "V"
    elif pattern == (0, 1, 1, 0, 1): return "W"
    elif pattern == (1, 1, 0, 0, 1): return "X"
    elif pattern == (1, 0, 0, 0, 1): return "Y"
    elif pattern == (0, 1, 1, 0, 1) and z_diff_index < 0.03: return "Z"
    return None

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7,
                    min_tracking_confidence=0.7,
                    max_num_hands=2) as hands:
    text_buffer = ""
    last_letter = ""
    last_time = 0
    delay = 1.5  # seconds

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                letter = recognize_letter(hand_landmarks.landmark)

                current_time = time.time()
                if letter and letter != last_letter and current_time - last_time > delay:
                    text_buffer += letter
                    last_letter = letter
                    last_time = current_time

        cv2.putText(frame, f'Text: {text_buffer}', (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Sign Language Writer", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break
        elif key == ord('c'):  # 'c' to clear screen
            text_buffer = ""
            last_letter = ""

cap.release()
cv2.destroyAllWindows()
