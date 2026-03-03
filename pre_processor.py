from deepface import DeepFace
from PIL import Image
import os
import numpy as np
import pillow_heif  # Import here
import cv2

pillow_heif.register_heif_opener()  # MUST be early!

detector_backend = 'retinaface'  # or 'mediapipe' for ultra-fast, 'retinaface' for max accuracy

input_dir = 'data/raw_photos'
output_dir = 'data/family_photos'

#img = Image.open("data/raw_photos/raja/2025-04-08_04-53-05_114.heic")
#print("HEIC opened successfully:", img.size)
#img.convert("RGB").save("test.jpg")

for person in os.listdir(input_dir):
    if person.startswith('.') or not os.path.isdir(os.path.join(input_dir, person)):
        continue  # Skip .DS_Store, hidden files, stray files, etc.
    person_dir = os.path.join(input_dir, person)
    out_person_dir = os.path.join(output_dir, person)
    os.makedirs(out_person_dir, exist_ok=True)
    for img_file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_file)
        try:
            pil_img = Image.open(img_path)
            pil_img = pil_img.convert("RGB")  # Ensure RGB
            
            # Convert to numpy array – DeepFace works well with RGB or BGR
            img_array = np.array(pil_img)  # shape (H, W, 3), uint8, RGB

            # Extract faces using chosen backend
            faces = DeepFace.extract_faces(
                img_path=img_array,
                detector_backend=detector_backend,
                enforce_detection=False,  # Don't raise error if no face
                align=True  # Auto-align for better DeiT input
            )
            if faces:
                print(f"Detected faces in {img_file}")
                for i, face in enumerate(faces):
                    face_array = face['face']
                    # Debug: Check what we got
                    #print(f"Face {i} from {img_file}: shape={face_array.shape}, dtype={face_array.dtype}, "
                          #f"min={face_array.min()}, max={face_array.max()}")
                    
                    # Scale if it's float normalized [0,1] (RetinaFace sometimes does this)
                    if face_array.dtype in [np.float32, np.float64] and face_array.max() <= 1.0 + 1e-6:
                        face_array = (face_array * 255).clip(0, 255).astype(np.uint8)
                        #print("Scaled float to uint8")
                    
                    # FIX: Convert BGR → RGB (safe even if already RGB, but crucial for RetinaFace)
                    if face_array.shape[-1] == 3:
                        face_array = cv2.cvtColor(face_array, cv2.COLOR_BGR2RGB)
                        #print("Applied BGR → RGB conversion")
                    
                    # Now create and save
                    cropped_img = Image.fromarray(face_array)
                    base_name = os.path.splitext(img_file)[0]
                    save_path = os.path.join(out_person_dir, f"{base_name}_face{i}.jpg")
                    cropped_img.save(save_path)
                    print(f"Saved faces in {save_path}")
            else:
                print(f"No faces in {img_file}")
        except Exception as e:
            print(f"Error processing {img_file}: {e}")