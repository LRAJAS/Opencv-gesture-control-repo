import cv2
import time

# Initialize the webcam
cam = cv2.VideoCapture(0)

# Variables to calculate FPS
prev_frame_time = 0
new_frame_time = 0

# Loop to get frames
while True:
    # Read the frame from the camera
    success, img = cam.read()
    
    # Check if the frame was successfully read
    if not success:
        break

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    
    # Convert FPS to an integer
    fps = int(fps)
    
    # Convert FPS to string
    fps = str(fps)
    
    # Put FPS text on the frame
    cv2.putText(img, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Image', img)
    
    # Wait for 1 millisecond before moving to the next frame
    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all OpenCV windows
cam.release()
cv2.destroyAllWindows()
