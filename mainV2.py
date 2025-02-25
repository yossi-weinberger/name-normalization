from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from throttler import throttle
import asyncio
import csv
import itertools
import json
import os
import re
from usage_tracker import tracker
from token_limiter import limiter

load_dotenv()

# DATASET_PATH = "./lp_members.csv"
DATASET_PATH = "./test_cases.csv"
RESULT_DIR_PATH = "./results/"
TASKS_PATH = "./tasks.jsonl"

client = OpenAI()
async_client = AsyncOpenAI()

"""
מערכת להפרדת שמות עבריים לשם פרטי ושם משפחה
משתמש ב-OpenAI API כדי לנתח שמות מקובץ CSV ולייצר פלט מאורגן

קבצי קלט/פלט:
- קובץ קלט: CSV עם שמות לעיבוד
- קובץ פלט: CSV עם השמות המופרדים
- קובץ משימות: JSONL עם בקשות API מוכנות
"""

# קבועים לתבנית התשובה מ-OpenAI API
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "name_schema",
        "schema": {
            "type": "object",
            "required": [
                "is_person",
                "first_name",
                "last_name",
                "confidence",
                "name_order",
            ],
            "properties": {
                "is_person": {
                    "type": "boolean",
                    "description": "האם זה שם של אדם",
                },
                "first_name": {
                    "type": "string",
                    "description": "השם הפרטי",
                },
                "last_name": {
                    "type": "string",
                    "description": "שם המשפחה",
                },
                "confidence": {
                    "type": "number",
                    "description": "רמת הביטחון בניתוח (0-1)",
                },
                "name_order": {
                    "type": "string",
                    "enum": ["first_last", "last_first"],
                    "description": "סדר השמות במקור",
                },
            },
            "additionalProperties": False,
        },
        "strict": True,
    },
}

PROMPT = """
Analyze the given Hebrew text and determine:
1. Whether it is a person's name (not a business, institution, etc.).
2. Identify the first name and last name.
3. Determine whether the name appears in regular order (first-last) or reversed (last-first).

### Rules for Identifying Person Names:
1. **Indicators of a Business/Organization (NOT a person's name):**
   - Contains terms such as: בע"מ, בעמ, ltd, בע״מ.
   - Starts with words like: מסעדת, חנות, מרקט, מאפיית, בית.
   - Multiple business-related words appearing together.

2. **Valid Person Name Patterns:**
   - [First] [Last]: "יוסי כהן"
   - [Last] [First]: "רוזנבלט יוסי"
   - [Title] [Last] [First]: "ד"ר שפירא ויקטוריה"
   - Hyphenated names: "כהן-לוי", "בן-גוריון"

3. **Special Cases:**
   - Titles (ד"ר, עו"ד, פרופ') should be removed, but the name should still be marked as a valid person's name.
   - Hyphenated names should be kept as a single unit.
   - Names with prefixes such as בן-, בת-, אבו-, בר- are considered part of the last name.

### Output Requirements:
1. The output must use exactly the same Hebrew characters as in the input.
2. No non-Hebrew characters should appear in the output.
3. The output should only include words that were present in the input.
"""


@throttle(rate_limit=500, period=60)
async def process_name(name: str) -> tuple[str, str, float | None]:
    """
    מעבד שם בודד באמצעות OpenAI API
    
    Args:
        name: השם המלא לעיבוד
        
    Returns:
        tuple המכיל את השם הפרטי, שם המשפחה ורמת הביטחון
    """
    try:
        # הערכת מספר הטוקנים והמתנה אם צריך
        estimated_tokens = limiter.estimate_tokens(name)
        await limiter.consume(estimated_tokens)
        
        cleaned_name = preprocess_name(name)
        
        response = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": cleaned_name}
            ],
            response_format=RESPONSE_FORMAT
        )

        # הדפסת מידע על הטוקנים
        print(f"\033[36m[Tokens] {name}: estimated={estimated_tokens}, actual={response.usage.total_tokens}\033[0m")  # כחול בהיר
        
        # עדכון לפי השימוש בפועל
        await limiter.adjust_tokens(
            estimated_tokens,
            response.usage.total_tokens
        )
        
        # הוספת מעקב שימוש
        tracker.add_usage(response, model="gpt-4o-mini")
        
        result = json.loads(response.choices[0].message.content)
        
        if result["is_person"] and result["confidence"] > 0.6:
            first_name = result["first_name"]
            last_name = result["last_name"]
            confidence = result["confidence"]
            
            if not (first_name and last_name):
                return (None, None, None)
                
            if not is_name_valid(first_name) or not is_name_valid(last_name):
                return (None, None, None)
                
            return (first_name, last_name, confidence)
        else:
            return (None, None, None)
    except Exception as e:
        print(f"Error processing name: {name}, {e}")
        return (None, None, None)

def validate_name(name: str) -> str | None:
    """
    בודק אם השם תקין לעיבוד
    מחזיר None אם תקין, או סיבת דחייה אם לא תקין
    """
    # שם חייב להכיל לפחות 2 מילים
    if len(name.split()) < 2:
        return "single_word"
        
    # שם לא יכול להכיל מילים באנגלית
    if re.search(r'[a-zA-Z]', name):
        return "contains_english"
        
    # שם לא יכול להכיל מילים כמו "בע"מ"
    if re.search(r'בע"מ|בעמ|ltd|בע״מ', name.lower()):
        return "company_name"
        
    # שם לא יכול להכיל מספרים
    if re.search(r'\d', name):
        return "contains_numbers"
        
    return None

def preprocess_name(name: str) -> str:
    """מעבד את השם לפני שליחה ל-API"""
    # הסרת תארים - רשימה מורחבת
    titles = [
        r'ד"ר',
        r'הרב',
        r"ד'ר",
        r'דר\'',
        r'דר',
        r'עו"ד',
        r'עוה"ד',
        r'רו"ח',
        r'פרופ\'',
        r'פרופ',
        r'ד״ר',  # תו גרש אחר
        r'רופ\'ש',  # מקרה מיוחד שראינו בדאטה
    ]
    
    # מסיר את כל התארים
    for title in titles:
        name = re.sub(rf'{title}\s+', '', name)
    
    # חיבור שמות עם מקף
    name = re.sub(r'(בן|בת|אבו)\s+([א-ת]+)', r'\1-\2', name)
    
    return name.strip()

def postprocess_results(name_parts: list, parts: dict) -> tuple[str, str]:
    """מעבד את התוצאות מה-API"""
    first_names = []
    last_names = []
    
    for i, part in enumerate(name_parts):
        if parts[f"part_{i}"] == "G":
            first_names.append(part)
        else:  # F
            # אם זה חלק משם משפחה מורכב
            if i > 0 and parts[f"part_{i-1}"] == "F":
                last_names[-1] = f"{last_names[-1]}-{part}"
            else:
                last_names.append(part)
    
    return " ".join(first_names), " ".join(last_names)

def validate_output(first_name: str, last_name: str, original_text: set) -> bool:
    """בודק שכל המילים בפלט נמצאות בקלט"""
    if not first_name or not last_name:
        return False
        
    # מנקה את המילים מתווים מיוחדים ומפריד לפי רווחים
    output_words = set()
    for word in (first_name + " " + last_name).split():
        # מנקה מקפים ותווים מיוחדים
        cleaned_word = word.replace('-', ' ').replace('"', '').replace("'", '')
        output_words.update(cleaned_word.split())
    
    # מנקה גם את המילים המקוריות באותה צורה
    cleaned_original = set()
    for word in original_text:
        cleaned_word = word.replace('-', ' ').replace('"', '').replace("'", '')
        cleaned_original.update(cleaned_word.split())
    
    return output_words.issubset(cleaned_original)

def is_hebrew_only(text: str) -> bool:
    """בודק שהטקסט מכיל אותיות עבריות בלבד (ומקף ורווח)"""
    return bool(re.match(r'^[א-ת\s\-]+$', text))

async def process(stop: int, start: int = 0) -> None:
    """
    מעבד קבוצת שמות מתוך קובץ הקלט

    Args:
        stop: אינדקס הסיום לעיבוד
        start: אינדקס ההתחלה לעיבוד (ברירת מחדל: 0)
    """
    # איפוס המעקב בתחילת הריצה
    tracker.reset()
    
    result_path = os.path.join(RESULT_DIR_PATH, f"{start:05d}-{stop:05d}V2.csv")
    reject_path = os.path.join(RESULT_DIR_PATH, f"{start:05d}-{stop:05d}V2_rejects.csv")

    with (
        open(DATASET_PATH, "r", encoding="utf-8-sig") as src,
        open(result_path, "w", newline="", encoding="utf-8") as dst,
        open(reject_path, "w", newline="", encoding="utf-8") as reject_dst,
    ):
        reader = itertools.islice(csv.DictReader(src), start, stop)
        writer = csv.DictWriter(dst, fieldnames=["id", "first_name", "last_name", "original_order", "confidence"])
        reject_writer = csv.DictWriter(reject_dst, fieldnames=["id", "name", "reason"])

        writer.writeheader()
        reject_writer.writeheader()

        chunk_size = 50
        while True:
            chunk = list(itertools.islice(reader, chunk_size))
            if not chunk:
                break

            tasks = [
                (row["CUST"], row["CUSTDES"], process_name(row["CUSTDES"]))
                for row in chunk if is_name_valid(row["CUSTDES"])
            ]
            
            responses = await asyncio.gather(*[task[2] for task in tasks])
            
            for (id, full_name), response in zip([(t[0], t[1]) for t in tasks], responses):
                if response[0] is None or response[1] is None:
                    reject_writer.writerow({
                        "id": id,
                        "name": full_name,
                        "reason": "not_a_person" if not response[0] else "processing_error"
                    })
                    continue

                writer.writerow({
                    "id": id,
                    "first_name": response[0],
                    "last_name": response[1],
                    "original_order": "first_last" if full_name.startswith(response[0]) else "last_first",
                    "confidence": response[2]
                })

            dst.flush()
            reject_dst.flush()
            
    # הדפסת סיכום בסוף הריצה
    tracker.print_summary()

def is_name_valid(name: str) -> bool:
    """
    בודק אם השם מכיל רק תווים עבריים חוקיים ותווים מיוחדים מותרים

    Args:
        name: השם לבדיקה

    Returns:
        True אם השם תקין, False אחרת
    """
    # מאפשר גם מקף, גרשיים כפולים ורווחים
    return not re.search(r"[^א-תפףךםןץ\s'\"-]", name)

def generate_tasks(limit: int = None) -> None:
    """
    מייצר קובץ JSONL עם משימות API מוכנות
    
    Args:
        limit: מגביל את מספר השורות לעיבוד (אופציונלי)
    """
    with open(DATASET_PATH, "r", encoding="utf-8-sig") as src, \
         open(TASKS_PATH, "w", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        if limit:
            reader = itertools.islice(reader, limit)
            
        for row in reader:
            if not is_name_valid(row["CUSTDES"]):
                continue
                
            task = {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": row["CUSTDES"]}
                ]
            }
            dst.write(json.dumps(task) + "\n")

def create_batch() -> None:
    """יוצר batch job חדש ב-OpenAI API"""
    batch_job = client.batches.create(
        file=TASKS_PATH,
        completion_config={
            "model": "gpt-4",
            "max_tokens": 150
        }
    )
    print(batch_job)

def get_batch(batch_id: str) -> None:
    """
    מקבל תוצאות של batch job
    
    Args:
        batch_id: המזהה של ה-batch job
    """
    batch_job = client.batches.retrieve(batch_id)
    print(batch_job)
    
    if batch_job.status == "completed":
        print("Batch is completed")
        content = client.files.content(batch_job.output_file_id).content
        with open("results.jsonl", "wb") as f:
            f.write(content)

def print_names_with_special_characters() -> None:
    """מדפיס שמות שמכילים תווים מיוחדים"""
    with open(DATASET_PATH, "r", encoding="utf-8-sig") as src:
        reader = csv.DictReader(src)
        counter = 0
        for row in reader:
            if not is_name_valid(row["CUSTDES"]):
                print(row["CUSTDES"])
                counter += 1
        print(f"Total names with special characters: {counter}")

if __name__ == "__main__":
    asyncio.run(process(5, 0))
    # generate_tasks(1000)
    # create_batch()
    # get_batch("batch_67b8ecac035c8190a4365678412c89ad")
    # print_names_with_special_characters()
