import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  # Change to 1 if you want to use the second camera
    detector = htm.handDetector()

    while True:
        success, img = cap.read()
        if not success:
            break
        
        img = detector.findHands(img, draw=True)
        lmList = detector.findPosition(img, draw=False)

        if lmList:
            print(lmList[4])  # Print the coordinates of landmark index 4

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
