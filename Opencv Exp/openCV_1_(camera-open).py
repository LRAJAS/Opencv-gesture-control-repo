import cv2

# Initialize the webcam
cam = cv2.VideoCapture(0)

# Loop to continuously get frames
while True:
    # Read the frame from the camera
    success, img = cam.read()
    
    # Check if the frame was successfully read
    if not success:
        break
    
    # Display the frame
    cv2.imshow('Image', img)
    
    # Wait for 1 millisecond before moving to the next frame
    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all OpenCV windows
cam.release()
cv2.destroyAllWindows()
