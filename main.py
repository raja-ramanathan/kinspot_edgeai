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

SAVE_DIR="model"
CUSTOM_MODEL_NAME='kinspotmodel'
MODEL_NAME='google/vit-base-patch16-224'
DEVICE = torch.device('mps' if torch.mps.is_available() else 'cpu')

class KinSpotModel:

    def __init__(self):
        # HYPER-PARAMETERS
        self.class_names = ["anuja", "raja"]
        self.num_classes = len(self.class_names)  # Number of family/friends
        self.batch_size = 16
        self.epochs = 5
        self.learning_rate = 1e-4
        self.processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std),
        ])
        # Load pre-trained DeiT and modify head
        self.model = ViTForImageClassification.from_pretrained(MODEL_NAME,
            num_labels=self.num_classes,ignore_mismatched_sizes=True  # Allows replacing the head
        )
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

        # Save model
        self.model.save_pretrained(f"model/{CUSTOM_MODEL_NAME}")

    def _infer(self):
        pass

    def process(self):
        if not self._is_model_saved():
            print(f"Model NOT found in {SAVE_DIR} → Start training")
            self._train()
        self._infer()


def main():
    kinSpotModel = KinSpotModel()
    kinSpotModel.process()


if __name__ == "__main__":
    main()
