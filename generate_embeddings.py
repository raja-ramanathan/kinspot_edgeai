import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import faiss

from config import TRAINING_DIR, MODEL_EMBEDDING_DIR
from face_embedding import FaceEmbeddingExtractor
from main import KinspotModelLoader, DEVICE
from utils import get_valid_labels, get_valid_files


class FaceEmbeddingGenerator:
    """ Generates mean face embeddings for each label in the training dataset."""

    def __init__(self):
        self.embedding_extractor = FaceEmbeddingExtractor()
        self. label_db = {}
        self.label_map = {}
        self.index = None

    def _generate_embeddings(self, label_dir):
        """ Generate mean embedding for a given label directory."""
        embeddings = []
        mean_embedding = None
        for img_file in get_valid_files(label_dir):
            img_path = os.path.join(label_dir, img_file)
            try:
                print("processing file...", img_path)
                img = Image.open(img_path)
                img = img.convert("RGB")  # Ensure RGB
                emb = self.embedding_extractor.extract_image_embedding(img)
                embeddings.append(emb)
                mean_embedding = torch.mean(torch.cat(embeddings), dim=0)
                mean_embedding = F.normalize(mean_embedding, dim=0)
            except Exception as e:
                raise e
        return mean_embedding

    def build_faiss_index(self):
        """Build FAISS index from embeddings."""

        vectors = []
        labels = []

        for idx, (label, emb) in enumerate(self.label_db.items()):
            vec = emb.cpu().numpy().astype("float32")

            vectors.append(vec)
            labels.append(label)

            self.label_map[idx] = label

        vectors = np.vstack(vectors)

        dim = vectors.shape[1]

        # Using cosine similarity
        index = faiss.IndexFlatIP(dim)

        index.add(vectors)

        self.index = index

        print("FAISS index built")
        print("Total vectors:", index.ntotal)

    def generate_embeddings(self, training_dir):
        """ Generate mean embeddings for all labels in the training directory."""
        for label in get_valid_labels(training_dir):
            label_dir = os.path.join(training_dir, label)
            print("processing label_dir...", label_dir)
            mean_embedding = self._generate_embeddings(label_dir)
            self.label_db[label] = mean_embedding

    def save_embeddings(self, embeddings_dir):
        os.makedirs(embeddings_dir, exist_ok=True)
        torch.save(self.label_db, f"{embeddings_dir}/kinspot_embeddings.pt")
        # Save FAISS index
        faiss.write_index(self.index, f"{embeddings_dir}/faiss_index.bin")
        # Save label map
        with open(f"{embeddings_dir}/label_map.json", "w") as f:
            json.dump(self.label_map, f)
        print("Saved FAISS index and label map")

def main():
    embedding_generator = FaceEmbeddingGenerator()
    embedding_generator.generate_embeddings(TRAINING_DIR)
    embedding_generator.build_faiss_index()
    embedding_generator.save_embeddings(MODEL_EMBEDDING_DIR)
        
if __name__ == "__main__":
    main()