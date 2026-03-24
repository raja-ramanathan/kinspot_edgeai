import cv2
import torch
from torch.nn.functional import cosine_similarity

from config import DEVICE, CONFIDENCE_THRESHOLD
from face_recognition import FaceDetector, FaceRecognitionSystem
from main import KinspotModelLoader
from face_embedding import FaceEmbeddingExtractor

class FacePredictor:
    """ Predicts the identity of a face using Open-Set model."""
    def __init__(self):
        self.confidence_threshold = CONFIDENCE_THRESHOLD  # adjust as needed
        self.embedding_extractor = FaceEmbeddingExtractor()
        self.label_db_file = "embeddings/kinspot_embeddings.pt"
        self.label_db = torch.load(self.label_db_file)

    def _find_best_match(self,embedding):
        best_score = -1
        best_name = None
        for name, db_emb in self.label_db.items():
            score = cosine_similarity(embedding, db_emb.unsqueeze(0))
            if score > best_score:
                best_score = score
                best_name = name
        return best_name, best_score.item()

    def predict_face(self, face_img):
            embedding = self.embedding_extractor.extract_face_embedding(face_img)
            best_name, best_score = self._find_best_match(embedding)
            if best_score > self.confidence_threshold:
                return best_name, best_score
            else:
                return "Not a family member", best_score

def main():
    face_predictor = FacePredictor()
    face_recognition_system = FaceRecognitionSystem(face_predictor)
    face_recognition_system.start()
    face_recognition_system.shutdown()

if __name__ == "__main__":
    main()