import torch
import torch.nn.functional as F
from config import DEVICE
from main import KinspotModelLoader
from torchvision import transforms

class FaceEmbeddingExtractor:
    """ Extracts face embeddings using the Custom ViT model."""
    def __init__(self):
        self.modelLoader = KinspotModelLoader()
        self.model = self.modelLoader.model
        self.processor = self.modelLoader.processor
        self.face_transform = self.modelLoader.transform
        self.image_transform = self.modelLoader.imageTransform

    def _extract_embedding(self, input_tensor):
        with torch.no_grad():
            outputs = self.model.vit(pixel_values=input_tensor)
            embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def extract_face_embedding(self,face_img):
        input_tensor = self.face_transform(face_img).unsqueeze(0).to(DEVICE)
        return self._extract_embedding(input_tensor)

    def extract_image_embedding(self, face_img):
        input_tensor = self.image_transform(face_img).unsqueeze(0).to(DEVICE)
        return self._extract_embedding(input_tensor)

