import cv2
import mediapipe as mp
import time

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for the default webcam

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.75)

# Get drawing utilities
mp_draw = mp.solutions.drawing_utils

# Variables for FPS calculation
p_time = 0

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Convert BGR to RGB (MediaPipe requires RGB format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform face detection
    results = face_detection.process(img_rgb)

    if results.detections:
        ih, iw, _ = img.shape  # Get image dimensions once
        for detection in results.detections:
            # Get bounding box coordinates
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                          int(bboxC.width * iw), int(bboxC.height * ih))
            
            # Draw bounding box and confidence score
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    # Calculate and display FPS
    c_time = time.time()
    fps = int(1 / (c_time - p_time))
    p_time = c_time
    cv2.putText(img, f'FPS: {fps}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the output
    cv2.imshow("Face Detection", img)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
