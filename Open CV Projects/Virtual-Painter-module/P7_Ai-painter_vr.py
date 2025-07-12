import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

# Constants
w_cam, h_cam = 640, 480  # Webcam dimensions
frame_reduction = 100  # Frame reduction for better mouse control
smoothening = 7  # Smoothening factor for mouse movement

# Initialize variables
p_time = 0  # Previous time for FPS calculation
ploc_x, ploc_y = 0, 0  # Previous location of the mouse
cloc_x, cloc_y = 0, 0  # Current location of the mouse

# Initialize webcam
cap = cv2.VideoCapture(1)
cap.set(3, w_cam)
cap.set(4, h_cam)

# Initialize hand detector
detector = htm.HandDetector(max_hands=1)

# Get screen size
w_scr, h_scr = autopy.screen.size()

while True:
    # 1. Find hand landmarks
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    img = detector.find_hands(img)
    lm_list, bbox = detector.find_position(img)

    # 2. Get the tip of the index and middle fingers
    if lm_list:
        x1, y1 = lm_list[8][1:]  # Index finger tip
        x2, y2 = lm_list[12][1:]  # Middle finger tip

        # 3. Check which fingers are up
        fingers = detector.fingers_up()
        cv2.rectangle(img, (frame_reduction, frame_reduction), 
                     (w_cam - frame_reduction, h_cam - frame_reduction), 
                     (255, 0, 255), 2)

        # 4. Only Index Finger: Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert coordinates to screen size
            x3 = np.interp(x1, (frame_reduction, w_cam - frame_reduction), (0, w_scr))
            y3 = np.interp(y1, (frame_reduction, h_cam - frame_reduction), (0, h_scr))

            # 6. Smoothen values for better mouse movement
            cloc_x = ploc_x + (x3 - ploc_x) / smoothening
            cloc_y = ploc_y + (y3 - ploc_y) / smoothening

            # 7. Move mouse
            autopy.mouse.move(w_scr - cloc_x, cloc_y)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            ploc_x, ploc_y = cloc_x, cloc_y

        # 8. Both Index and Middle Fingers Up: Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find distance between fingers
            length, img, line_info = detector.find_distance(8, 12, img)
            print(length)

            # 10. Click mouse if distance is short
            if length < 40:
                cv2.circle(img, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # 11. Calculate and display FPS
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. Display the image
    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()