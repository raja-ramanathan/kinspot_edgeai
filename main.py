import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from torch.optim import AdamW 
from tqdm import tqdm  # For progress bars
from sklearn.metrics import accuracy_score  # For evaluation
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from utils import get_valid_labels

from pillow_heif import register_heif_opener #to support HEIC files

register_heif_opener() #to support HEIC files

# Model names
CUSTOM_MODEL_NAME='kinspotmodel'
MODEL_NAME='google/vit-base-patch16-224'

# MODEL data directories
SAVE_DIR=Path(f"model/{CUSTOM_MODEL_NAME}")
TEST_DIR=Path("data/test") # testing dataset dir
TRAIN_DIR=Path("data/family_photos/train") # training dataset dir
VAL_DIR=Path("data/family_photos/train") # validation dataset dir

# Use GPU for mac
DEVICE = torch.device('mps' if torch.mps.is_available() else 'cpu')


class KinSpotModel:

    def __init__(self):
        # HYPER-PARAMETERS
        self.class_names = list(get_valid_labels(TRAIN_DIR))
        self.id2label = {i: label for i, label in enumerate(self.class_names)}
        self.label2id = {val: key for key, val in self.id2label.items()}
        self.num_classes = len(self.id2label)  # Number of family/friends
        self.batch_size = 16
        self.epochs = 5
        self.learning_rate = 1e-4
        self.processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std),
        ])
        # Load saved or pre-trained mode.
        if self._is_model_saved():
            self.model = ViTForImageClassification.from_pretrained(SAVE_DIR)
        else:
            self.model = ViTForImageClassification.from_pretrained(
                MODEL_NAME,num_labels=self.num_classes,
                id2label=self.id2label,
                label2id=self.label2id,
               ignore_mismatched_sizes=True) 
        self.model.to(DEVICE)
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

    def _is_model_saved(self) -> bool:
        if not os.path.isdir(SAVE_DIR):
            return False
        
        files = set(os.listdir(SAVE_DIR))
        # Check for the essentials (safetensors is now standard)
        has_weights = "model.safetensors" in files or "pytorch_model.bin" in files
        has_config  = "config.json" in files
    
        return has_weights and has_config
        
    def _train(self):
        # Load custom dataset (like ImageFolder for CIFAR but custom)
        train_dataset = datasets.ImageFolder(root='data/family_photos/train', transform=self.transform)  # Split your data into train/val
        val_dataset = datasets.ImageFolder(root='data/family_photos/val', transform=self.transform)

        #train_dataset = Subset(train_dataset, range(5))
        #val_dataset = Subset(val_dataset, range(5))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Training loop (similar to your CIFAR setup)
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for images, labels in tqdm(train_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images).logits
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss / len(train_loader):.4f}")

        # Evaluation
        self.model.eval()
        preds, true_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                outputs = self.model(images).logits
                preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(true_labels, preds)
        print(f"Validation Accuracy: {acc:.4f}")

        # print confusion matrix with evaluation
        confmat = ConfusionMatrix(num_classes=self.num_classes, task='multiclass')
        confmat_tensor = confmat(preds=torch.tensor(preds),
                             target=torch.tensor(true_labels))
        fig, ax = plot_confusion_matrix(
            conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy 
            class_names=self.class_names, # turn the row and column labels into class names
            figsize=(10, 7)
          );
        out_path = Path("output/vit.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)  

        # Save model & processor
        self.model.save_pretrained(SAVE_DIR)
        self.processor.save_pretrained(SAVE_DIR)

    def _predict_single_image(self,image_path):
        # Open image
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening {image_path}: {e}")
            return None, None

        # Preprocess (resize to 224×224, normalize, etc.)
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = inputs.to(DEVICE)

        # prepare model in eval model
        kinSpotModel = ViTForImageClassification.from_pretrained(SAVE_DIR)
        kinSpotModel.to(DEVICE)
        kinSpotModel.eval() 

        # Run model
        with torch.inference_mode():
            outputs = kinSpotModel(**inputs)
            logits = outputs.logits  # [1, 10]
            probabilities = torch.softmax(logits, dim=-1)[0]  # [10]
            predicted_class_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class_idx].item()
            print(logits, probabilities, predicted_class_idx, confidence)
        predicted_label = self.class_names[predicted_class_idx]
        return predicted_label, confidence


    def _test(self):
        """ Infer & test using selective images in test directory """
        for person in os.listdir(TEST_DIR):
            if person.startswith('.') or not os.path.isdir(os.path.join(TEST_DIR, person)):
                continue  # Skip .DS_Store, hidden files, stray files, etc.
            person_dir = os.path.join(TEST_DIR, person)
            for img_file in os.listdir(person_dir):
                test_image_path = os.path.join(person_dir, img_file)
                print("======")
                print(f"Processing file: {test_image_path} for testing...")
                try:
                    predicted_label, confidence = self._predict_single_image(test_image_path)
                    print("expected_label=", person, "predicted_label=", predicted_label, " confidence=", confidence)
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")

    def process(self):
        # train when saved model is not available
        if not self._is_model_saved():
            print(f"Model NOT found in {SAVE_DIR} → Start training")
            self._train()
        else:
            print(f"Saved model found in {SAVE_DIR} → Skip training")        
        self._test()
        

class KinspotModelLoader:
    """ Load the model, processor and id2label from saved model directory """
    def __init__(self):
        self.model = ViTForImageClassification.from_pretrained(SAVE_DIR)
        self.processor = ViTImageProcessor.from_pretrained(SAVE_DIR)
        self.id2label = self.model.config.id2label
        self.model.to(DEVICE)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(), #face to image
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std)
        ])
        self.imageTransform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std)
        ])

def main():
    kinSpotModel = KinSpotModel()
    kinSpotModel.process()

if __name__ == "__main__":
    main()
