import cv2
import time

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces

class FaceRecognitionSystem:
    def __init__(self, face_predictor=None):
        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        time.sleep(5)
        self.face_predictor = face_predictor
        self.face_detector = FaceDetector()

    def start(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            faces = self.face_detector.detect_faces(frame)
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                label, confidence = self.face_predictor.predict_face(face)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame,
                            f"{label} ({confidence:.2f})",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,255,0),
                            2)

            cv2.imshow("Family Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("exiting....")
                break

    def shutdown(self):
        self.cap.release()
        cv2.destroyAllWindows()
