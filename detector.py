# from sklearn.datasets import load_iris
import pandas as pd
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import cv2
import mediapipe as mp
import pandas as pd
import customUtils
from sklearn import preprocessing
le = preprocessing.LabelEncoder()


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

data = pd.read_csv("Data/initialTests.csv")
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
le.fit(y)
y=le.transform(y)

# neigh = SVR(n_neighbors=3)
neigh = SVC()
neigh.fit(X, y)

cap = cv2.VideoCapture(0)
cordinate_list = []
counter = {}

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2,
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
            cordinates = []
            for handResult in results.multi_hand_landmarks:
                
                x_min, y_min, z_min, x_max, y_max, z_max = customUtils.findMinMax(handResult.landmark, w, h)
                for landmark in handResult.landmark:
                    cordinates.append(landmark.x - x_min)
                    cordinates.append(landmark.y - y_min)
                    cordinates.append(landmark.z - z_min)
                cv2.rectangle(
                    frame,
                    (int(x_min * w), int(y_min * h)),
                    (int(x_max * w), int(y_max * h)),
                    (0, 255, 0),
                    1,
                )
                mp_drawing.draw_landmarks(
                    frame,
                    handResult,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        cv2.imshow("Input", frame)
        c = cv2.waitKey(1)

        try:
            if c == 27:
                break
            if c >= 32:
                if len(cordinates)<125:
                    while len(cordinates)<=125:
                        cordinates.append(0)
                pred_class = neigh.predict([cordinates])
                print(f"Class : {pred_class}")
        except:
            print(
                "SORRY COULDNT LOG COORDINATES! IDK WHY SO ASK CJ AS HE IS SMARTER THAN YOU!"
            )

    cap.release()
    cv2.destroyAllWindows()

# data = load_iris()

#  X = pd.DataFrame(data=data.data, columns=data.feature_names)
# y = pd.DataFrame(data=data.target, columns=["Class"])
# classes = list(data.target_names)


pred_class = neigh.predict([[6.0, 3.4, 4.5, 1.6]])
print(pred_class[0])
