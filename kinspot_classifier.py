import cv2
import torch
from face_recognition import FaceDetector, FaceRecognitionSystem
from main import KinspotModelLoader
from config import DEVICE, CONFIDENCE_THRESHOLD

class FacePredictor:
    """Predicts the identity of a face using a Closed Set model."""
    def __init__(self):
        self.confidence_threshold = CONFIDENCE_THRESHOLD  # adjust as needed
        self.modelLoader = KinspotModelLoader()
        self.id2label = self.modelLoader.id2label
        self.model = self.modelLoader.model
        self.transform = self.modelLoader.transform

    def predict_face(self, face_img):
        face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(face).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = self.model(pixel_values=input_tensor)
            logits = outputs.logits  # [1, 10]
            probabilities = torch.softmax(logits, dim=-1)[0]  # [10]
            predicted_class_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class_idx].item()
            print(logits, probabilities, predicted_class_idx, confidence)

        label = self.id2label[predicted_class_idx] if confidence > self.confidence_threshold else "Not a family member"
        return label, confidence


def main():
    face_predictor = FacePredictor()
    face_recognition_system = FaceRecognitionSystem(face_predictor)
    face_recognition_system.start()
    face_recognition_system.shutdown()

if __name__ == "__main__":
    main()