import cv2
import mediapipe as mp
import pandas as pd


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
cordinate_list = []
counter = {}

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1,
) as hands:

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        cordinates = []
        ret, frame = cap.read()

        h, w, c = frame.shape

        if not ret:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame_bgr.flags.writeable = False
        results = hands.process(frame_bgr)
        frame_bgr.flags.writeable = True
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for handResult in results.multi_hand_landmarks:
                cordinates = []
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h

                z_min = "blah"
                for landmark in handResult.landmark:
                    x, y, z = int(landmark.x * w), int(landmark.y * h), int(landmark.z)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                    if z_min == "blah" or z < z_min:
                        z_min = z

                    cordinates.append(landmark.x - x_min)
                    cordinates.append(landmark.y - y_min)
                    cordinates.append(landmark.z - z_min)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        cv2.imshow("Input", frame)

        c = cv2.waitKey(1)
        try:
            if c == 27:
                break
            if c >= 97 and c <= 122:
                if c in counter:
                    counter[c] += 1
                else:
                    counter[c] = 1
                cordinates.append(chr(c))
                print(f"You pressed {chr(c)} {counter[c]} times!")
                cordinate_list.append(cordinates)
        except:
            print(
                "SORRY COULDNT LOG COORDINATES! IDK WHY SO ASK CJ AS HE IS SMARTER THAN YOU!"
            )

    df = pd.DataFrame.from_records(cordinate_list)
    df.to_csv("./Data/initialTests.csv")
    cap.release()
    cv2.destroyAllWindows()
