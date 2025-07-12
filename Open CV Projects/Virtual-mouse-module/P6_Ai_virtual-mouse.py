import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

# Parameters
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 7

# Variables
pTime = 0
plocX, plocY = 0, 0  # Previous location
clocX, clocY = 0, 0  # Current location

# Capture Video
cap = cv2.VideoCapture(1)
cap.set(3, wCam)  # Width
cap.set(4, hCam)  # Height

# Initialize Hand Detector
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()  # Get screen dimensions
# print(wScr, hScr)

while True:
    # 1. Capture Frame and Detect Hands
    success, img = cap.read()
    if not success:
        print("Failed to read frame from camera.")
        break

    img = detector.findHands(img)  # Detect hand
    lmList, bbox = detector.findPosition(img)  # Get landmark positions

    # 2. Check for Landmark Availability
    if lmList:
        # Get coordinates of index (8) and middle (12) fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # Draw boundary box for interaction
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 4. Moving Mode: Only Index Finger Up
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert Coordinates to Screen Space
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # 6. Smooth Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 7. Move Mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY  # Update previous location

        # 8. Clicking Mode: Index and Middle Fingers Up
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find Distance Between Fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # print(length)

            # 10. Perform Click if Distance is Short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # 11. Frame Rate Calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. Display Output
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
        break

cap.release()
cv2.destroyAllWindows()
