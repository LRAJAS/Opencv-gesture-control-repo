import cv2
import mediapipe as mp
import time

class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        """
        Initialize the Face Mesh Detector.
        
        Parameters:
        - staticMode: Whether to treat input images as static (False for video).
        - maxFaces: Maximum number of faces to detect.
        - minDetectionCon: Minimum confidence for face detection.
        - minTrackCon: Minimum confidence for tracking.
        """
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        # Initialize MediaPipe Face Mesh
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            staticMode=self.staticMode, 
            max_num_faces=self.maxFaces, 
            min_detection_confidence=self.minDetectionCon, 
            min_tracking_confidence=self.minTrackCon
        )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)  # Drawing style

    def findFaceMesh(self, img, draw=True):
        """
        Detect facial landmarks and draw the face mesh.
        
        Parameters:
        - img: Input image (frame from video or image).
        - draw: Whether to draw the mesh on the face.
        
        Returns:
        - Processed image with mesh (if draw=True).
        - List of facial landmarks (x, y coordinates).
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB (required by MediaPipe)
        results = self.faceMesh.process(imgRGB)  # Process the image
        faces = []

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec
                    )

                face = []
                for lm in faceLms.landmark:
                    ih, iw, _ = img.shape  # Get image dimensions
                    x, y = int(lm.x * iw), int(lm.y * ih)  # Convert normalized coordinates
                    face.append([x, y])
                faces.append(face)

        return img, faces

def main():
    cap = cv2.VideoCapture("Videos/1.mp4")  # Load video file or use 0 for webcam
    pTime = 0  # Previous time for FPS calculation
    detector = FaceMeshDetector(maxFaces=2)  # Create an instance of FaceMeshDetector

    while cap.isOpened():  # Check if video is successfully opened
        success, img = cap.read()
        if not success:
            break  # Exit loop if no frame is captured

        img, faces = detector.findFaceMesh(img)

        if faces:
            print(f"Face landmarks detected: {len(faces[0])} points")  # Print number of landmarks

        # Calculate FPS (Frames Per Second)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Display FPS on screen
        cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow("Face Mesh Detection", img)  # Show video frame

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
