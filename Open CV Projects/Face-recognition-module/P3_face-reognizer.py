import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("face_recognizer.h5")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img_resized = cv2.resize(img, (100, 100))
    img_array = np.expand_dims(img_resized, axis=0) / 255.0

    predictions = model.predict(img_array)
    label = np.argmax(predictions)

    cv2.putText(img, f"Person: {label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
