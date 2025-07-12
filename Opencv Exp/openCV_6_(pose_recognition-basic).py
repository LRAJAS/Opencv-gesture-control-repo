import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Pose and Drawing Utilities
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Capture Video
cap = cv2.VideoCapture('Pose1.mp4')
pTime = 0

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Convert the BGR image to RGB for MediaPipe processing
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        # Draw pose landmarks on the original frame
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        # Print landmark coordinates and draw circles on specific points
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            print(id, cx, cy)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    # Calculate and display FPS on the frame
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    # Display the processed frame
    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
