import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define subjects and additional fields
subjects = ["Maths", "Science", "English", "History", "Computer"]
genders = ["Male", "Female"]
grades = [9, 10, 11, 12]

# Function to generate a more realistic student record
def generate_student_record(student_id):
    # Demographics
    gender = random.choice(genders)
    grade = random.choice(grades)
    internet = random.choices(["Yes", "No"], weights=[0.8, 0.2])[0]

    # Attendance
    attendance = int(np.clip(np.random.normal(85, 10), 50, 100))

    # Base ability influenced by attendance
    base_ability = np.clip(np.random.normal(70, 15), 30, 100)
    attendance_influence = (attendance - 75) * 0.3  # ±15 boost/penalty
    base_ability = np.clip(base_ability + attendance_influence, 30, 100)

    # Generate subject marks with variation and correlation
    marks = {
        "Maths": int(np.clip(base_ability + np.random.normal(0, 5), 30, 100)),
        "Science": int(np.clip(base_ability + np.random.normal(0, 5), 30, 100)),
        "English": int(np.clip(base_ability + np.random.normal(-5, 7), 30, 100)),
        "History": int(np.clip(base_ability + np.random.normal(-5, 7), 30, 100)),
        "Computer": int(np.clip(base_ability + np.random.normal(5, 10), 30, 100)),
    }

    # Occasionally introduce an anomaly (5% chance)
    if random.random() < 0.05:
        random_subject = random.choice(subjects)
        anomaly = random.choice([-20, 20])
        marks[random_subject] = int(np.clip(marks[random_subject] + anomaly, 30, 100))

    # Calculate average marks
    avg_marks = sum(marks.values()) / len(subjects)

    # Recommendation logic
    if avg_marks >= 75 and attendance >= 85:
        recommendation = "Eligible for Advanced Courses"
    elif avg_marks >= 50:
        recommendation = "Needs Improvement"
    else:
        recommendation = "High Risk of Failure"

    return {
        "StudentID": f"S{student_id:05}",
        "Gender": gender,
        "Grade": grade,
        "Internet Access": internet,
        **marks,
        "Attendance(%)": attendance,
        "Recommendation": recommendation
    }

# Parameters
total_rows = 100000
target_per_category = total_rows // 3
MAX_ATTEMPTS = total_rows * 10  # safety cap to avoid infinite loop

counts = {
    "Eligible for Advanced Courses": 0,
    "Needs Improvement": 0,
    "High Risk of Failure": 0
}

data = []
student_id = 1
attempts = 0

# Generate data with balanced recommendations
while sum(counts.values()) < total_rows and attempts < MAX_ATTEMPTS:
    student = generate_student_record(student_id)
    rec = student["Recommendation"]

    if counts[rec] < target_per_category:
        data.append(student)
        counts[rec] += 1
        student_id += 1

    attempts += 1

# Create DataFrame and save
df = pd.DataFrame(data)
df.to_csv("balanced_student_data.csv", index=False)

print("✔️ Realistic and balanced dataset created: 'balanced_student_data.csv'")
print("📊 Final category counts:", counts)
print("🔁 Total attempts made:", attempts)
