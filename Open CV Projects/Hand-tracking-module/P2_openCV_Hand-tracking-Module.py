import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  # Change to 1 if you want to use the second camera
    detector = htm.handDetector()
    draw_color = (255, 0, 255)  # Color for drawing
    xp, yp = 0, 0  # Previous points for drawing

    while True:
        success, img = cap.read()
        if not success:
            break
        
        img = detector.findHands(img, draw=True)
        lmList = detector.findPosition(img, draw=False)

        if lmList:
            # Get the position of the index finger tip (landmark id 8)
            x, y = lmList[8][1], lmList[8][2]

            # Draw on the screen
            if xp == 0 and yp == 0:
                xp, yp = x, y
            
            # Draw a line from the previous point to the current point
            cv2.line(img, (xp, yp), (x, y), draw_color, 15)
            xp, yp = x, y

        # Calculate and display FPS (optional)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Uncomment to display FPS on the image
        # cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
