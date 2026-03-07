# Real-Time Edge Face Recognition with Vision Transformers

Trained and deployed a Vision Transformer for multi-person classification using a custom dataset. Implemented real-time camera inference, face detection pre-processing, and GPU-accelerated inference on macOS (Metal) with planned TensorRT deployment on NVIDIA Jetson for embedded edge optimization.


## Project Setup

- Clone the repo
- uv sync

##  Data Directory Setup
  Gather photos for the persons and organize as below...
```
data/ <- overall dataset folder
    train/ <- training images
        mom/ <- class name as folder name
            image01.jpeg
            image02.jpeg
            ...
        dad/
            image24.jpeg
            image25.jpeg
            ...
        bro/
            image37.jpeg
            ...
    validation/ <- validating images
        mom/
            image101.jpeg
            image102.jpeg
            ...
        dad/
            image154.jpeg
            image155.jpeg
            ...
        bro/
            image167.jpeg
            ...
     test/  <- test images (ideally not related to training/validation)
        mom/
            image201.jpeg
            image202.jpeg
            ...
        dad/
            image254.jpeg
            image255.jpeg
            ...
        bro/
            image267.jpeg
            ...
```
## Train/Validation/Testing
 Run the below command to train, validate and save the model. Once the model is saved, it is tested using the test images. Review the test results for accuracy.  

- uv run main.py

## Edge AI Deployment
 Requires camera for this. Run the command below to leverage the model for identifying the person. 

- uv run kinspot.py 


