# Advanced AI Medical Intelligence Platform

## 1. Introduction
This project is an AI-powered medical image analysis system that predicts Pneumonia from Chest X-ray images using a Convolutional Neural Network (CNN). The application provides a user-friendly web interface for image upload, prediction, confidence score, and prediction history.

## 2. Objectives
- Detect Pneumonia from Chest X-ray images.
- Display prediction confidence.
- Store prediction history.
- Provide REST APIs.
- Build a complete AI web application.

## 3. Technologies Used
- Python
- TensorFlow
- Flask
- SQLite
- HTML
- CSS
- NumPy
- Pillow

## 4. System Architecture
User Upload → Flask → CNN Model → Prediction → Database → Display Result

## 5. Features
- Chest X-ray Upload
- Pneumonia Detection
- Confidence Score
- Image Preview
- Prediction History
- REST API

## 6. Model
A custom CNN model with:
- Conv2D Layers
- MaxPooling Layers
- Dense Layer
- Dropout Layer
- Sigmoid Output

## 7. Database
SQLite database stores:
- Image Name
- Prediction
- Date & Time

## 8. API
- /api/health
- /api/predict

## 9. Results
The model predicts:
- NORMAL
- PNEUMONIA

along with confidence score.

## 10. Conclusion
The project successfully demonstrates an AI-based medical image analysis system using Deep Learning and Flask.