import cv2
import torch
import time
from torch.nn.functional import cosine_similarity
from main import KinspotModelLoader, DEVICE
from torchvision import transforms
import torch.nn.functional as F

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
time.sleep(5)

confidence_threshold = 0.85  # adjust as needed

modelLoader = KinspotModelLoader()
model = modelLoader.model
processor = modelLoader.processor
transform = modelLoader.transform
id2label = modelLoader.id2label
label_db_file = "embeddings/kinspot_embeddings.pt"
label_db = torch.load(label_db_file)

def extract_embedding(face_img):
    input_tensor = transform(face_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model.vit(pixel_values=input_tensor)
        embedding = outputs.last_hidden_state[:, 0, :]   # CLS token
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding

def find_best_match(embedding):

    best_score = -1
    best_name = None

    for name, db_emb in label_db.items():
        score = cosine_similarity(embedding, db_emb.unsqueeze(0))
        if score > best_score:
            best_score = score
            best_name = name
    return best_name, best_score.item()

def main():
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
            embedding = extract_embedding(face)
            label, confidence = find_best_match(embedding)
            print(label,confidence)
            if not (confidence > confidence_threshold):
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