# Real-Time Edge Face Recognition with Vision Transformers

Trained and deployed a Vision Transformer for multi-person classification using a custom dataset. Implemented real-time camera inference, face detection pre-processing, and GPU-accelerated inference on macOS (Metal) with planned TensorRT deployment on NVIDIA Jetson for embedded edge optimization.


## Project Setup

- Clone the repo
- uv sync

##  Data Directory Setup

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
    validation/ <- testing images
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

## Train/Validation/Testing

- uv run main.py

