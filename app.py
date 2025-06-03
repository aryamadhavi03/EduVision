from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import easyocr
from PIL import Image
import cv2
import numpy as np
import re
import os
import pickle
import pandas as pd
import logging
from pdf2image import convert_from_path

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs('static/css', exist_ok=True)
os.makedirs('templates', exist_ok=True)

app = Flask(__name__)
app.secret_key = 'EduVision'  # Required for session; replace with a secure key

# Load the trained model and LabelEncoder
try:
    with open("student_recommendation_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    logger.info("Model and LabelEncoder loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model or LabelEncoder: {e}")
    raise

# Initialize OCR reader
try:
    reader = easyocr.Reader(['en'], gpu=False)
    logger.info("OCR reader initialized.")
except Exception as e:
    logger.error(f"Failed to initialize OCR reader: {e}")
    raise

def preprocess_image(image):
    """Preprocess image for better OCR accuracy."""
    try:
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Corrected typo
                                      cv2.THRESH_BINARY, 11, 2)
        scale_percent = 200
        width = int(thresh.shape[1] * scale_percent / 100)
        height = int(thresh.shape[0] * scale_percent / 100)
        resized = cv2.resize(thresh, (width, height), interpolation=cv2.INTER_LINEAR)
        denoised = cv2.fastNlMeansDenoising(resized)
        return Image.fromarray(denoised)
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise

def extract_text_from_file(image):
    """Extract text from image using EasyOCR."""
    try:
        img = preprocess_image(image)
        results = reader.readtext(np.array(img), detail=0)
        text = "\n".join(results).strip()
        logger.debug(f"Extracted text: {text}")
        return text
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        return f"Error extracting text: {e}"

def extract_marks_from_text(text):
    """Extract subject-wise marks and attendance from text."""
    try:
        subject_patterns = {
            "Computer": r"\b(?:Computer|Comp|Computer\s*Science)\b(?!\w)",
            "Maths": r"\b(?:Math|Mathematics|Mathcmatics)\b",
            "Science": r"\b(?:Science|Scicnce|SCENCE)\b(?!\s*(STREAM|Computer))",
            "English": r"\b(?:English|Eng|Eoglish)\b",
            "History": r"\b(?:History|Hist)\b"
        }
        number_pattern = r"^[0-9]{1,3}(?:\.[0-9]+)?$"
        
        marks = {subject: None for subject in subject_patterns}
        lines = text.split("\n")
        in_academic_section = False
        last_subject = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if "PERFORMANCE" in line.upper():
                in_academic_section = True
                continue
            if not in_academic_section:
                continue
            
            for subject, pattern in subject_patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    last_subject = subject
                    for j in range(i + 1, min(i + 6, len(lines))):
                        next_line = lines[j].strip()
                        if re.match(number_pattern, next_line):
                            try:
                                mark = float(next_line)
                                if 0 <= mark <= 100 and marks[subject] is None:
                                    marks[subject] = mark
                                    last_subject = None
                                    break
                            except ValueError:
                                continue
                    break

        attendance_pattern = r"Attendance[\s:-]*([0-9]{1,3}(?:\.[0-9]+)?)\s*(?:/|%)?"
        attendance_match = re.search(attendance_pattern, text, re.IGNORECASE)
        attendance = float(attendance_match.group(1)) if attendance_match else None

        logger.debug(f"Extracted marks: {marks}, attendance: {attendance}")
        return marks, attendance
    except Exception as e:
        logger.error(f"Marks extraction failed: {e}")
        raise

@app.route('/', methods=['GET'])
def home():
    """Render a simple HTML form for uploading images."""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Failed to render template: {e}")
        return jsonify({"error": "Template rendering failed"}), 500

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    """Handle image upload, OCR, prediction, and result display."""
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                logger.error("No file part in request")
                return jsonify({"error": "No file part"}), 400
            
            file = request.files['file']
            if file.filename == '':
                logger.error("No selected file")
                return jsonify({"error": "No selected file"}), 400
            
            if file:
                file_path = "temp_file"
                file.save(file_path)
                logger.debug(f"File saved as {file_path}")
                
                # Handle PDF or image
                if file.filename.lower().endswith('.pdf'):
                    images = convert_from_path(file_path)
                    image = images[0]  # Use the first page
                else:
                    image = Image.open(file_path)
                
                extracted_text = extract_text_from_file(image)
                
                if "Error" in extracted_text:
                    return jsonify({"error": extracted_text}), 500
                
                marks, attendance = extract_marks_from_text(extracted_text)
                
                features = ["Maths", "Science", "English", "History", "Computer", "Attendance(%)"]
                input_data = [marks.get(subject, 0) for subject in features[:-1]] + [attendance or 0]
                
                if any(v is None for v in input_data):
                    return jsonify({"error": "Missing or invalid data extracted"}), 400
                
                input_df = pd.DataFrame([input_data], columns=features)
                prediction = model.predict(input_df)
                prediction_proba = model.predict_proba(input_df)[0] 
                predicted_label = label_encoder.inverse_transform(prediction)[0]

                # Find the confidence of the predicted class
                predicted_class_index = prediction[0]
                confidence = prediction_proba[predicted_class_index] * 100  # Convert to percentage
                confidence = round(confidence, 2)
                
                os.remove(file_path)
                logger.info(f"Prediction successful: {predicted_label}")
                
                # Store data in session
                session['prediction_data'] = {
                    'marks': marks,
                    'attendance': attendance,
                    'prediction': predicted_label,
                    'confidence': confidence,
                    'chart_data': {
                        'labels': list(marks.keys()),
                        'values': [float(marks.get(subject, 0)) for subject in marks]
                    },
                    'attendance_data': {
                        'value': float(attendance or 0),
                        'remaining': 100 - float(attendance or 0)
                    }
                }
                
                # Redirect to the same route with GET
                return redirect(url_for('predict'))

        elif request.method == 'GET':
            # Retrieve data from session for GET request
            prediction_data = session.get('prediction_data')
            if not prediction_data:
                return jsonify({"error": "No prediction data available"}), 400
            
            return render_template('result.html',
                                marks=prediction_data['marks'],
                                attendance=prediction_data['attendance'],
                                prediction=prediction_data['prediction'],
                                confidence=prediction_data['confidence'],
                                chart_data=prediction_data['chart_data'],
                                attendance_data=prediction_data['attendance_data'])

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)