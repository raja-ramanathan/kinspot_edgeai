import cv2
import torch
import numpy as np
import time
from main import KinspotModelLoader, DEVICE

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
time.sleep(5)

confidence_threshold = 0.85  # adjust as needed
modelLoader = KinspotModelLoader()

def main():
    id2label = modelLoader.id2label
    model = modelLoader.model 
    transform = modelLoader.transform
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
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
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            input_tensor = transform(face).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(pixel_values=input_tensor)
                logits = outputs.logits  # [1, 10]
                probabilities = torch.softmax(logits, dim=-1)[0]  # [10]
                predicted_class_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class_idx].item()
                print(logits, probabilities, predicted_class_idx, confidence)

            if confidence > confidence_threshold:
                label = id2label[predicted_class_idx]
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

if __name__ == "__main__":
    main()