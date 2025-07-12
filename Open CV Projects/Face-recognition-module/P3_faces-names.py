import cv2
import numpy as np
import tensorflow as tf
import json

# Load the trained model
model = tf.keras.models.load_model("face_recognizer.h5")

# Load label mapping (Assuming it was saved as a JSON file during training)
with open("label_mapping.json", "r") as f:
    label_dict = json.load(f)  # Example: {"0": "Alice", "1": "Bob"}

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    # Preprocess the image
    img_resized = cv2.resize(img, (100, 100))  # Resize to match training size
    img_array = np.expand_dims(img_resized, axis=0) / 255.0  # Normalize

    # Get prediction
    predictions = model.predict(img_array)
    label_index = np.argmax(predictions)  # Get the index of the highest probability

    # Get the corresponding name from label_dict
    person_name = label_dict.get(str(label_index), "Unknown")  # Default to "Unknown"

    # Display the name on the image
    cv2.putText(img, f"Person: {person_name}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
