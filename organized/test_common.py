"""Test script for common modules functionality."""
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

from common.config import PROMPT, RESPONSE_FORMAT, RESULT_DIR_PATH
from common.schema import create_name_schema
from common.validators import is_name_valid, validate_name_parts
from common.writers import setup_writers, write_status_log

def test_name_validation():
    """Test name validation functions."""
    print("\nבדיקת וולידציה של שמות:")
    test_names = [
        "משה כהן",  # תקין
        "דר משה כהן",  # תקין
        "משה כהן123",  # לא תקין - מספרים
        "משה & כהן",  # לא תקין - תווים מיוחדים
    ]
    
    for name in test_names:
        is_valid = is_name_valid(name)
        print(f"השם '{name}' {'תקין' if is_valid else 'לא תקין'}")

def test_name_parts_validation():
    """Test name parts validation."""
    print("\nבדיקת וולידציה של חלקי שם:")
    test_cases = [
        ("משה כהן", "משה", "כהן"),  # תקין
        ("משה דוד כהן", "משה דוד", "כהן"),  # תקין
        ("משה כהן", "משה", "לוי"),  # לא תקין - שם משפחה לא קיים במקור
    ]
    
    for full_name, given, family in test_cases:
        is_valid, message = validate_name_parts(full_name, given, family)
        print(f"מקרה בדיקה: {full_name} -> {given} {family}")
        print(f"תוצאה: {'תקין' if is_valid else 'לא תקין'}")
        if not is_valid:
            print(f"הודעת שגיאה: {message}")

def test_schema_creation():
    """Test schema creation."""
    print("\nבדיקת יצירת סכמה:")
    test_name = "משה דוד כהן"
    schema = create_name_schema(test_name)
    print(f"סכמה שנוצרה עבור השם '{test_name}':")
    print(json.dumps(schema, indent=2, ensure_ascii=False))

def test_writers():
    """Test file writers."""
    print("\nבדיקת כתיבה לקבצים:")
    
    # וודא שהתיקייה קיימת
    if not os.path.exists(RESULT_DIR_PATH):
        os.makedirs(RESULT_DIR_PATH)
    
    # בדיקת setup_writers
    writers = setup_writers(0, 10, RESULT_DIR_PATH)
    result_writer, reject_writer, result_file, reject_file = writers
    
    try:
        # כתיבת נתוני בדיקה
        result_writer.writerow({
            "id": "001",
            "given_name": "משה",
            "family_name": "כהן"
        })
        
        reject_writer.writerow({
            "id": "002",
            "name": "משה & כהן",
            "reason": "invalid_characters"
        })
        
        print("נכתבו נתוני בדיקה לקבצי התוצאות והדחיות")
        
        # בדיקת כתיבת לוג
        write_status_log(
            batch_id="test_batch_123",
            batch_status="completed",
            request_counts={"completed": 1, "failed": 1, "total": 2},
            usage_data={
                "total_calls": 2,
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "total_cost": 0.002
            },
            range_str="00000-00010",
            result_dir_path=RESULT_DIR_PATH
        )
        print("נכתב קובץ לוג")
        
    finally:
        # סגירת הקבצים
        result_file.close()
        reject_file.close()

def main():
    """Run all tests."""
    print("בדיקת פונקציונליות המודולים המשותפים:")
    print("=" * 50)
    
    test_name_validation()
    print("=" * 50)
    
    test_name_parts_validation()
    print("=" * 50)
    
    test_schema_creation()
    print("=" * 50)
    
    test_writers()
    print("=" * 50)
    
    print("\nכל הבדיקות הושלמו!")

if __name__ == "__main__":
    load_dotenv()
    main() 