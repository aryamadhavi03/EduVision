import easyocr
from PIL import Image
import cv2
import numpy as np
import re
import os

def preprocess_image(image):
    """Preprocess image for better OCR accuracy."""
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    scale_percent = 200  # Increased to improve clarity
    width = int(thresh.shape[1] * scale_percent / 100)
    height = int(thresh.shape[0] * scale_percent / 100)
    resized = cv2.resize(thresh, (width, height), interpolation=cv2.INTER_LINEAR)
    denoised = cv2.fastNlMeansDenoising(resized)
    return Image.fromarray(denoised)

def extract_text_from_file(file_path):
    """Extract text from image using EasyOCR."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        
        reader = easyocr.Reader(['en'], gpu=False)
        img = Image.open(file_path)
        img = preprocess_image(img)
        results = reader.readtext(np.array(img), detail=0)
        return "\n".join(results).strip()
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def extract_marks_from_text(text):
    """Extract subject-wise marks and attendance from text."""
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
        # Detect academic section (relaxed pattern)
        if "PERFORMANCE" in line.upper():
            in_academic_section = True
            print(f"Debug: Entered academic section at line: {line}")
            continue
        if not in_academic_section:
            continue
        
        # Check for subject
        for subject, pattern in subject_patterns.items():
            if re.search(pattern, line, re.IGNORECASE):
                print(f"Debug: Found subject {subject} in line: {line}")
                last_subject = subject
                # Look ahead for mark
                for j in range(i + 1, min(i + 6, len(lines))):  # Increased to 6 lines
                    next_line = lines[j].strip()
                    if re.match(number_pattern, next_line):
                        try:
                            mark = float(next_line)
                            if 0 <= mark <= 100 and marks[subject] is None:
                                marks[subject] = mark
                                print(f"Debug: Assigned mark {mark} to {subject} from line: {next_line}")
                                last_subject = None  # Reset after assignment
                                break
                        except ValueError:
                            continue
                break

    # Extract attendance
    attendance_pattern = r"Attendance[\s:-]*([0-9]{1,3}(?:\.[0-9]+)?)\s*(?:/|%)?"
    attendance_match = re.search(attendance_pattern, text, re.IGNORECASE)
    attendance = float(attendance_match.group(1)) if attendance_match else None
    if attendance:
        print(f"Debug: Extracted attendance {attendance} from text")

    return marks, attendance

# Example usage
file_path = "image.png"  # Replace with your file path
extracted_text = extract_text_from_file(file_path)
if extracted_text:
    print("Extracted Text:\n", extracted_text)
    marks, attendance = extract_marks_from_text(extracted_text)
    print("\nExtracted Marks:", {k: v for k, v in marks.items() if v is not None})
    print("Attendance:", attendance)
else:
    print("No text extracted. Check file or OCR setup.")