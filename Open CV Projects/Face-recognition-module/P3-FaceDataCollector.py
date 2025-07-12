import cv2
import os
import mediapipe as mp

class FaceDataCollector:
    def __init__(self, save_path="faces", num_images=10):
        """Initialize face dataset collector with storage path and image count."""
        self.save_path = save_path
        self.num_images = num_images
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    def collect_faces(self):
        """Captures faces from webcam, prompts for name, and saves images."""
        cap = cv2.VideoCapture(0)

        while True:
            success, img = cap.read()
            if not success:
                break

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.detector.process(img_rgb)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = img.shape
                    x, y, w, h = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                                  int(bboxC.width * iw), int(bboxC.height * ih))

                    face = img[y:y+h, x:x+w]
                    face = cv2.resize(face, (100, 100))

                    # Ask for the user's name
                    name = input("Enter the name for this face: ").strip()
                    person_path = os.path.join(self.save_path, name)
                    
                    if not os.path.exists(person_path):
                        os.makedirs(person_path)

                    # Save images in the person's directory
                    count = len(os.listdir(person_path))
                    file_name = f"{person_path}/face_{count}.jpg"
                    cv2.imwrite(file_name, face)
                    print(f"Saved {file_name}")

            cv2.imshow("Face Collection", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Run the script
collector = FaceDataCollector()
collector.collect_faces()
