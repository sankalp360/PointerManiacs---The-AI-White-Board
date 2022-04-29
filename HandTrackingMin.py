import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0  # previous time
cTime = 0  # current time

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # MediaPipe Hands Only Uses RBG images, so we converted our pixel image into RGB
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmark)
    # 21 landmarks
    # information of hands with id no

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                print(id, lm);
                # height, width, channel of image
                h, w, c = img.shape
                # x, y position landmarks are in decimals we convert them into pixels
                cx, cy = int(lm.x * w), int(lm.y * h)


            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    fps = int(fps)
    pTime = cTime

    cv2.putText(img, str("FPS " + str(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
