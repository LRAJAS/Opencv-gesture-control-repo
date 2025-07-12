import cv2
import mediapipe as mp
import numpy as np
import math
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc

class FindHands:
    def __init__(self, detection_con=0.5, tracking_con=0.5):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(min_detection_confidence=detection_con, min_tracking_confidence=tracking_con)
        self.mpDraw = mp.solutions.drawing_utils
    
    def getPosition(self, img, indexes, hand_no=0, draw=True):
        lst = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) >= hand_no + 1:
                for id, lm in enumerate(results.multi_hand_landmarks[hand_no].landmark):
                    for index in indexes:
                        if id == index:
                            h, w, c = img.shape
                            x, y = int(lm.x * w), int(lm.y * h)
                            lst.append((x, y))
                if draw:
                    self.mpDraw.draw_landmarks(img, results.multi_hand_landmarks[hand_no], self.mpHands.HAND_CONNECTIONS)
        return lst

    def index_finger_up(self, img, hand_no=0):
        pos = self.getPosition(img, (6, 8), draw=False)
        if len(pos) == 2:
            return pos[0][1] >= pos[1][1]
        return "NO HAND FOUND"

    def middle_finger_up(self, img, hand_no=0):
        pos = self.getPosition(img, (10, 12), draw=False)
        if len(pos) == 2:
            return pos[0][1] >= pos[1][1]
        return "NO HAND FOUND"

    def ring_finger_up(self, img, hand_no=0):
        pos = self.getPosition(img, (14, 16), draw=False)
        if len(pos) == 2:
            return pos[0][1] >= pos[1][1]
        return "NO HAND FOUND"

    def little_finger_up(self, img, hand_no=0):
        pos = self.getPosition(img, (18, 20), draw=False)
        if len(pos) == 2:
            return pos[0][1] >= pos[1][1]
        return "NO HAND FOUND"

################################
wCam, hCam = 640, 480
################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
detector = FindHands(detection_con=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        continue

    img = detector.getPosition(img, indexes=[4, 8])
    lmList = detector.getPosition(img, indexes=[4, 8], draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[0]
        x2, y2 = lmList[1]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        # Hand range 50 - 300
        # Volume Range -65 - 0
        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])

        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        # Brightness control based on length
        brightness = np.interp(length, [50, 300], [0, 100])
        sbc.set_brightness(brightness)

    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Img", img)
    cv2.waitKey(1)
