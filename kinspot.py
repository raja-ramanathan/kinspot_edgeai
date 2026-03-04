import cv2
import torch
import numpy as np
from torchvision import transforms
from pathlib import Path
from transformers import ViTForImageClassification
import time


CUSTOM_MODEL_NAME='kinspotmodel'
SAVE_DIR=Path(f"model/{CUSTOM_MODEL_NAME}")
DEVICE = torch.device('mps' if torch.mps.is_available() else 'cpu')
kinSpotModel = ViTForImageClassification.from_pretrained(SAVE_DIR)
kinSpotModel.to(DEVICE)
kinSpotModel.eval()

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
print("Is opened:", cap.isOpened())
time.sleep(5)

confidence_threshold = 0.75  # adjust as needed

label_map = {
    0: "anuja",
    1: "raja",
}


def main():    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    print("Starting....")
    label = "UNKNOWN"
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            input_tensor = transform(face).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = kinSpotModel(pixel_values=input_tensor)
                logits = outputs.logits  # [1, 10]
                probabilities = torch.softmax(logits, dim=-1)[0]  # [10]
                predicted_class_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class_idx].item()
                print(logits, probabilities, predicted_class_idx, confidence)

            if confidence > confidence_threshold:
                label = label_map[predicted_class_idx]
            else:
                label = "Not a family member"

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

    cap.release()
    cv2.destroyAllWindows()

    print("label is", label)

if __name__ == "__main__":
    main()