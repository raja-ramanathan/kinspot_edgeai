# Real-Time Edge Face Recognition with Vision Transformers

This project implements a real-time face recognition system using Vision Transformers (ViTs) optimized
for edge devices, specifically NVIDIA Jetson platforms. The system is designed to recognize family members 
in real-time using a camera feed, leveraging the power of ViTs for accurate face recognition 
while maintaining efficiency on resource-constrained devices. 

The project includes data pre-processing, model training, and real-time inference components, 
making it a comprehensive solution for edge AI face recognition applications. It also demonstrates the 
use of TensorRT for optimizing the model inference, ensuring low latency and high performance.

Demonstrates both Open Set and Closed Set face recognition approaches, allowing for 
flexibility in handling known and unknown individuals in the recognition process.

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
## Pre-processing
 The pre-processing script `preprocess.py` detects faces in the images and crops them to focus on the face region. 
 This step is crucial for improving the accuracy of the Vision Transformer model by ensuring that it learns from 
 the relevant features of the images.

## Train/Validation/Testing
 Run the below command to train, validate and save the model. Once the model is saved, 
 it is tested using the test images. Review the test results for accuracy.  

- uv run main.py

## Edge AI Face Recognition
 You can run the real-time face recognition using the camera. The script `kinspot_*.py` captures video from the camera, 
 detects faces in real-time, and uses the trained Vision Transformer model to identify the person. 
This will start the camera and display the video feed with detected faces and their predicted labels using 
 TensorRT for optimized inference on NVIDIA Jetson devices.

- uv run kinspot_classifier.py -> Based on Closed Set model leveraging the classifier layer. 
- uv run kinspot_embeddings.py -> Based on Open Set model leveraging the embedding layer and cosine similarity for classification.

