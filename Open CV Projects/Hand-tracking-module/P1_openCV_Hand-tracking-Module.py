import cv2
import mediapipe as mp
import time

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=self.mode, 
                                         max_num_hands=self.maxHands, 
                                         min_detection_confidence=self.detectionCon,
                                         min_tracking_confidence=self.trackCon)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, handNo=0, draw=True):
        lm_list = []
        self.results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if self.results.multi_hand_landmarks:
            hand_landmarks = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))
                if draw and id in [0, 4, 8, 12, 16, 20]:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        
        return lm_list

def main():
    cam = cv2.VideoCapture(0)
    prev_frame_time = 0

    detector = HandDetector()

    while cam.isOpened():
        success, img = cam.read()
        if not success:
            break

        img = detector.find_hands(img)
        lm_list = detector.find_position(img)

        # Display the frame
        cv2.imshow('Image', img)
        
        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
