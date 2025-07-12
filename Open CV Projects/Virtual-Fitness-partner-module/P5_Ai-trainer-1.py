import cv2
import numpy as np
import time
import PoseModule as pm

# Initialize variables
cap = cv2.VideoCapture("AiTrainer/curls.mp4")
detector = pm.poseDetector()
count, dir, pTime = 0, 0, 0

# Video Processing Loop
while True:
    success, img = cap.read()
    if not success:
        break  # Exit if video ends
    
    img = cv2.resize(img, (1280, 720))
    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)
    
    if lmList:
        # Right Arm Angle
        angle = detector.findAngle(img, 12, 14, 16)
        
        # Progress Percentage and Bar Calculation
        per = np.interp(angle, (210, 310), (0, 100))
        bar = np.interp(angle, (220, 310), (650, 100))
        
        # Curl Logic
        color = (255, 0, 255)
        if per == 100 and dir == 0:
            count += 0.5
            dir = 1
            color = (0, 255, 0)
        elif per == 0 and dir == 1:
            count += 0.5
            dir = 0
            color = (0, 255, 0)

        # Draw Progress Bar
        cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)
        
        # Draw Count
        cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)
    
    # FPS Calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
    
    # Display Output
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
