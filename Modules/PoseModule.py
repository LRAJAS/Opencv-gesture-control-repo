import cv2
import mediapipe as mp
import time
import math

class PoseDetector:
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Initialize Mediapipe Pose module
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode, 
                                     smooth_landmarks=self.smooth, 
                                     min_detection_confidence=self.detectionCon, 
                                     min_tracking_confidence=self.trackCon)
        self.lmList = {}

    def findPose(self, img, draw=True):
        """Detect pose landmarks and optionally draw them on the image."""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        
        return img

    def findPosition(self, img, draw=True):
        """Return a dictionary of landmark positions {id: (x, y)}."""
        self.lmList = {}
        if self.results.pose_landmarks:
            h, w, _ = img.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList[id] = (cx, cy)
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        """Calculate the angle between three landmarks."""
        if p1 in self.lmList and p2 in self.lmList and p3 in self.lmList:
            x1, y1 = self.lmList[p1]
            x2, y2 = self.lmList[p2]
            x3, y3 = self.lmList[p3]
            
            # Calculate angle
            angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
            angle = angle + 360 if angle < 0 else angle
            
            if draw:
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
                cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
                for (x, y) in [(x1, y1), (x2, y2), (x3, y3)]:
                    cv2.circle(img, (x, y), 10, (0, 0, 255), cv2.FILLED)
                    cv2.circle(img, (x, y), 15, (0, 0, 255), 2)
                cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
            return angle
        return None


def main(source=0):
    cap = cv2.VideoCapture(source)
    detector = PoseDetector()
    pTime = time.time()

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Video ended or cannot be loaded.")
            break

        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)

        if 14 in lmList:
            cv2.circle(img, lmList[14], 15, (0, 0, 255), cv2.FILLED)

        # FPS Calculation
        cTime = time.time()
        fps = int(1 / (cTime - pTime))
        pTime = cTime
        cv2.putText(img, f'FPS: {fps}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        
        cv2.imshow("Pose Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(0)  # Use 0 for webcam, or replace with video file path