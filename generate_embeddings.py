import torch
import os
from PIL import Image
from main import KinspotModelLoader, DEVICE
import torch.nn.functional as F
from pathlib import Path
from utils import get_valid_labels, get_valid_files
from torchvision import transforms

data_dir = 'data/family_photos/train'
embeddings_dir= "embeddings"
modelLoader = KinspotModelLoader()
model = modelLoader.model
processor = modelLoader.processor
transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

def extract_embedding(face_img):
    input_tensor = transform(face_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model.vit(pixel_values=input_tensor)
        embedding = outputs.last_hidden_state[:, 0, :]   # CLS token
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding

def generate_embeddings(label_dir):
    embeddings = []
    mean_embedding = None
    for img_file in get_valid_files(label_dir):
        img_path = os.path.join(label_dir, img_file)
        try:
            print("processing file...", img_path)
            img = Image.open(img_path)
            img = img.convert("RGB")  # Ensure RGB
            emb = extract_embedding(img)
            embeddings.append(emb)
            mean_embedding = torch.mean(torch.cat(embeddings), dim=0)
            mean_embedding = F.normalize(mean_embedding, dim=0)
        except Exception as e:
            raise e
    return mean_embedding


def main():
    label_db = {}
    for label in get_valid_labels(data_dir):
        label_dir = os.path.join(data_dir, label)
        print("processing label_dir...", label_dir)
        mean_embedding = generate_embeddings(label_dir)
        label_db[label] = mean_embedding
    os.makedirs(embeddings_dir, exist_ok=True)
    torch.save(label_db, f"{embeddings_dir}/kinspot_embeddings.pt")
        
if __name__ == "__main__":
    main()