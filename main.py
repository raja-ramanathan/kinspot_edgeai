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

def main():
    # Hyperparameters (tune as needed)
    num_classes = 2  # Number of family/friends
    batch_size = 16
    epochs = 5
    learning_rate = 1e-4
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')

    # Data transforms (DeiT expects 224x224)
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    # Load custom dataset (like ImageFolder for CIFAR but custom)
    train_dataset = datasets.ImageFolder(root='data/family_photos/train', transform=transform)  # Split your data into train/val
    val_dataset = datasets.ImageFolder(root='data/family_photos/val', transform=transform)

    #train_dataset = Subset(train_dataset, range(5))
    #val_dataset = Subset(val_dataset, range(5))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load pre-trained DeiT and modify head
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=num_classes,
        ignore_mismatched_sizes=True  # Allows replacing the head
    )
    model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training loop (similar to your CIFAR setup)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images).logits
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(true_labels, preds)
    print(f"Validation Accuracy: {acc:.4f}")
    print("preds", preds)
    print("true_labels", true_labels)

    confmat = ConfusionMatrix(num_classes=num_classes, task='multiclass')
    confmat_tensor = confmat(preds=torch.tensor(preds),
                         target=torch.tensor(true_labels))
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy 
        class_names=["anuja","raja"], # turn the row and column labels into class names
        figsize=(10, 7)
      );
    out_path = Path("output/vit.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)  


    # Save model
    model.save_pretrained('model/fine_tuned_deit_family')


if __name__ == "__main__":
    main()
