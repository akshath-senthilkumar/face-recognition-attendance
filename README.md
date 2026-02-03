# Face Recognition Attendance System

This project implements a face recognition–based attendance system using a lightweight deep learning model. The system performs face recognition using cosine similarity, logs attendance with IST timestamps, and raises alerts when unknown individuals are detected for a sustained duration.

---

## 🚀 Features
- Lightweight face recognition using InsightFace (buffalo model)
- Embedding-based face matching using cosine similarity
- Attendance logging with IST timestamps
- Each known person logged only once per session
- Unknown / stranger detection with time-based alert (>5 seconds)
- Annotated output video generation with bounding boxes

---

## 🧠 Project Workflow

### 1. Training Phase
- Load labeled face image dataset (person-wise folders)
- Detect faces and extract embeddings using InsightFace
- Store embeddings and labels for inference

### 2. Inference / Attendance Phase
- Load stored embeddings and labels
- Process video input frame by frame
- Perform face recognition using cosine similarity
- Log attendance with IST timestamps
- Trigger alert if an unknown person stays in frame for more than 5 seconds

---

## 🗂️ Project Structure

---

## 🛠️ Tech Stack
- Python
- OpenCV
- InsightFace (buffalo model)
- NumPy
- Scikit-learn
- pytz
- Google Colab

---

## ⚙️ Setup Instructions

Install dependencies:
```bash
pip install insightface onnxruntime scikit-learn opencv-python pytz
labeled_faces/
├── Person1/
│   ├── img1.jpg
│   ├── img2.jpg
├── Person2/
│   ├── img1.jpg
