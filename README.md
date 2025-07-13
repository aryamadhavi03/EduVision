# 🎓 EduVision - Student Academic Recommendation System

This project is an **automated tool** that analyzes **student marksheets** (image or PDF) and provides personalized recommendations based on marks and attendance using **OCR** and **Machine Learning**.

---

## 📌 Overview

1️⃣ **Upload**: Users upload a marksheet (PNG/JPG/PDF).  
2️⃣ **Extract**: OCR (EasyOCR, OpenCV) reads text from the document.  
3️⃣ **Process**: Extracted text is cleaned and parsed (subject marks & attendance).  
4️⃣ **Predict**: A trained `RandomForestClassifier` predicts one of:  
   - Eligible for Advanced Courses  
   - Needs Improvement  
   - High Risk of Failure  
5️⃣ **Display**: Results are shown with charts in a dashboard.

---

## ✅ Features

- 📂 Image & PDF uploads  
- 🔍 OCR text extraction  
- 📈 Visual results: bar & pie charts  
- ⚙️ 98% accurate ML model  
- 🔑 Easy-to-use Flask web app

---

## 🧩 Tech Stack

| Tech | Use |
|------|------|
| Python 3.7+ | Language |
| Flask | Web server |
| EasyOCR, OpenCV | OCR text extraction |
| pdf2image, Poppler | PDF to image |
| pandas, numpy | Data handling |
| scikit-learn | ML model (RandomForestClassifier) |
| matplotlib, seaborn | Charts |
| HTML, CSS | UI templates |

---

## 🖼️ GUI Screenshots

### 🏠 Home Page
<img width="783" height="624" alt="image" src="https://github.com/user-attachments/assets/0cce99d3-b563-40fe-b332-fe1cfe915948" />

### ✅ Result Page
<img width="778" height="844" alt="image" src="https://github.com/user-attachments/assets/95747065-d857-4d2d-b673-dc4914471859" />

---

## 📊 Model Performance

- Algorithm: RandomForestClassifier  
- Accuracy: 98% on test data  
- ROC AUC: 1.00 for all classes  
- Example confusion matrix:
  <img width="801" height="692" alt="image" src="https://github.com/user-attachments/assets/40dcbeaf-bbd9-45db-a58d-997b25d323ab" />

---

## ⚙️ Prerequisites

- Python 3.7+  
- `pip install flask easyocr opencv-python pillow pdf2image pandas numpy scikit-learn matplotlib seaborn`
- **Poppler**:
  - Windows: [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases)
  - Linux: `sudo apt-get install poppler-utils`
- Model files: `student_recommendation_model.pkl`, `label_encoder.pkl`

---
## 🚀 Let’s Automate Student Success!
