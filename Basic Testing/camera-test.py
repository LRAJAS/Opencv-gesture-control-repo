import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Change 0 to 1 if you have multiple cameras

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera opened successfully.")

# Capture a frame to display
success, frame = cap.read()
if not success:
    print("Error: Could not read frame.")
else:
    # Display the frame
    cv2.imshow('Camera Test', frame)
    cv2.waitKey(0)  # Press any key to close the window

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
