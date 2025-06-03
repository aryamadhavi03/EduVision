# Student Academic Recommendation System

This project is an automated system designed to analyze student marksheets or academic reports (in image or PDF format) and provide recommendations based on their performance. Using Optical Character Recognition (OCR) and a pre-trained machine learning model, the system extracts subject-wise marks and attendance details, evaluates the student's academic standing, and generates a recommendation. The output is presented in a clear, dashboard-style interface, making it useful for teachers, parents, and students.

## Overview

The system follows a streamlined process:
1. **Upload**: Users upload a marksheet or academic report in image (e.g., PNG, JPG) or PDF format.
2. **Text Extraction**: OCR techniques extract text from the uploaded document.
3. **Data Processing**: The system identifies subject-wise marks and attendance percentages from the extracted text.
4. **Prediction**: A trained machine learning model analyzes the data and generates a recommendation (e.g., "Eligible for Advanced Courses", "Needs Improvement", or "High Risk of Failure").
5. **Display**: Results are shown in a user-friendly dashboard with charts and a highlighted recommendation.

This automation enhances efficiency and supports informed decision-making for academic improvement.

## Features
- Supports image and PDF uploads.
- Uses EasyOCR for text extraction and preprocessing with OpenCV.
- Employs a pre-trained model for personalized recommendations.
- Visualizes results with bar charts (subject marks) and pie charts (attendance).
- Provides confidence scores for predictions.

## Prerequisites
- Python 3.7+
- Required Python packages:
  - `flask`
  - `easyocr`
  - `opencv-python`
  - `pillow`
  - `pdf2image`
  - `pandas`
  - `numpy`
  - `scikit-learn` (for model loading)
- Poppler utility (for PDF processing with `pdf2image`):
  - Install on Windows: Download from [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases) and add to PATH.
  - Install on Ubuntu: `sudo apt-get install poppler-utils`.
- Trained model files: `student_recommendation_model.pkl` and `label_encoder.pkl`.
