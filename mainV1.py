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

DATASET_PATH = "./lp_members.csv"
# DATASET_PATH = "./test_cases.csv"
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
1. Is this a person's name (not a business, institution, etc.)
2. What is the first name and last name
3. Is the name in regular order (first-last) or reversed (last-first)

Key rules for identifying person names:
1. Business/Organization indicators (NOT person names):
   - Contains: בע"מ, בעמ, ltd
   - Starts with: מסעדת, חנות, מרקט, מאפיית, בית
   - Multiple business words together

2. Valid person name patterns:
   - [First] [Last]: "יוסי כהן"
   - [Last] [First]: "רוזנבלט יוסי"
   - [Title] [Last] [First]: "ד"ר שפירא ויקטוריה"
   - [Last] [First1] [First2]: "לוי משה דוד"
   - Hyphenated names: "כהן-לוי", "בן-גוריון", "בת-חן"

3. Common Israeli name patterns:
   - Arabic names: "אבו-חמד מוחמד", "עבד אל-רחמן"
   - Russian/Eastern European: "זסלבסקי אלכסנדר", names ending with וב/סקי/ביץ'
   - Compound names: "בן הרוש מקסים", "לי-און קרולין"

4. Special cases:
   - Remove titles (ד"ר, עו"ד, פרופ') but mark as valid person names
   - Keep hyphenated names as one unit
   - Names with prefixes (בן-, בת-, אבו-, בר-) are family names

Output must:
1. Use exactly the same Hebrew characters as input
2. Not contain any non-Hebrew characters
3. Only use words that appear in the input
"""

@throttle(rate_limit=500, period=60)
async def process_name_v2(id: str, name: str) -> tuple[str, str, str | None]:
    """
    מעבד שם בודד באמצעות OpenAI API
    מחזיר tuple של (שם פרטי, שם משפחה, סיבת דחייה)
    אם השם תקין, סיבת הדחייה תהיה None
    """
    try:
        # הערכת מספר הטוקנים והמתנה אם צריך
        estimated_tokens = limiter.estimate_tokens(name)
        await limiter.consume(estimated_tokens)
        
        # עיבוד מקדים של השם
        processed_name = preprocess_name(name)
        
        # בדיקת תקינות על השם המעובד
        validation_error = validate_name(processed_name)
        if validation_error:
            return (None, None, validation_error)

        name_parts = processed_name.split()
        original_text = set(processed_name.replace('"', '').replace('-', ' ').split())

        # יוצר מילון של properties לכל חלק בשם
        properties = {
            f"part_{i}": {
                "type": "string",
                "description": f"Part {i + 1} of the name: {part}",
                "enum": ["G", "F"],
            }
            for i, part in enumerate(name_parts)
        }
        
        response = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
                    Analyze the given Hebrew name and determine for each part if it's a given name (G) or family name (F).
                    
                    Special rules:
                    1. Hyphenated names are one unit: "בן-גוריון", "כהן-לוי", "בת-חן", "אבו-חמד"
                    2. Ignore titles: "ד"ר", "עו"ד", "פרופ'"
                    3. Three-part names: 
                       - "family given1 given2"
                       - "given1 given2 family"
                    4. Prefixes with hyphen are family names: "בן-", "בת-", "אבו-"
                    5. Output must use EXACTLY the same Hebrew characters as input
                    6. DO NOT use Arabic characters or any non-Hebrew characters
                    
                    Examples:
                    - "יוסי כהן" -> {"part_0": "G", "part_1": "F"}
                    - "רוזנבלט יוסי" -> {"part_0": "F", "part_1": "G"}
                    - "בן-גוריון דוד" -> {"part_0": "F", "part_1": "G"}
                    - "לוי משה דוד" -> {"part_0": "F", "part_1": "G", "part_2": "G"}
                    """,
                },
                {"role": "user", "content": processed_name},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "name_parts_schema",
                    "schema": {
                        "type": "object",
                        "required": [f"part_{i}" for i in range(len(name_parts))],
                        "properties": properties,
                        "additionalProperties": False,
                    }
                }
            },
            temperature=0.1,
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
        
        # מפענח את התשובה מ-JSON
        parts = json.loads(response.choices[0].message.content)
        
        # מעבד את התוצאות
        first_name, last_name = postprocess_results(name_parts, parts)
        
        # בדיקה שהפלט מוכל בקלט
        if not validate_output(first_name, last_name, original_text):
            return (None, None, "output_not_in_input")
            
        # בדיקה שהפלט בעברית בלבד
        if not is_hebrew_only(first_name) or not is_hebrew_only(last_name):
            return (None, None, "non_hebrew_chars")
        
        return (first_name, last_name, None)
        
    except Exception as e:
        error_str = str(e).lower()
        if "rate_limit" in error_str:
            # הדפסת התראה לקונסול
            print(f"\033[33m[Rate Limit] Retrying: {name}\033[0m")  # צהוב
            return None
        return (None, None, f"processing_error: {str(e)}")

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
    """בודק שהטקסט מכיל אותיות עבריות בלבד (ומקף, גרש ורווח)"""
    return bool(re.match(r'^[א-ת\s\-\']+$', text))  # הוספנו ' לתווים המותרים

async def process(stop: int, start: int = 0) -> None:
    """
    מעבד קבוצת שמות מתוך קובץ הקלט

    Args:
        stop: אינדקס הסיום לעיבוד
        start: אינדקס ההתחלה לעיבוד (ברירת מחדל: 0)
    """
    # איפוס המעקב בתחילת הריצה
    tracker.reset()
    
    result_path = os.path.join(RESULT_DIR_PATH, f"{start:05d}-{(start + stop):05d}v1.csv")
    reject_path = os.path.join(RESULT_DIR_PATH, f"{start:05d}-{(start + stop):05d}v1_rejects.csv")

    with (
        open(DATASET_PATH, "r", encoding="utf-8-sig") as src,
        open(result_path, "w", newline="", encoding="utf-8") as dst,
        open(reject_path, "w", newline="", encoding="utf-8") as reject_dst,
    ):
        reader = itertools.islice(csv.DictReader(src), start, stop)
        writer = csv.DictWriter(dst, fieldnames=["id", "first_name", "last_name", "original_order"])
        reject_writer = csv.DictWriter(reject_dst, fieldnames=["id", "name", "reason"])

        writer.writeheader()
        reject_writer.writeheader()

        chunk_size = 50
        # מעקב אחרי שמות שכבר בתור - שינוי מ-set לרשימה
        pending_names = []
        
        while True:
            chunk = list(itertools.islice(reader, chunk_size))
            if not chunk and not pending_names:  # מסיים רק אם אין גם בצ'אנק וגם בממתינים
                break

            # מוסיף את השמות הממתינים לצ'אנק הנוכחי
            chunk.extend(pending_names)
            pending_names.clear()

            tasks = [
                (row["CUST"], row["CUSTDES"], process_name_v2(row["CUST"], row["CUSTDES"]))
                for row in chunk if is_name_valid(row["CUSTDES"])
            ]

            responses = await asyncio.gather(*[task[2] for task in tasks])

            for (id, full_name), response in zip([(t[0], t[1]) for t in tasks], responses):
                if response is None:
                    # מוסיף לרשימת הממתינים
                    pending_names.append({"CUST": id, "CUSTDES": full_name})
                    continue
                
                first_name, last_name, reject_reason = response
                
                if reject_reason:
                    reject_writer.writerow({
                        "id": id,
                        "name": full_name,
                        "reason": reject_reason
                    })
                    continue

                writer.writerow({
                    "id": id,
                    "first_name": first_name,
                    "last_name": last_name,
                    "original_order": "first_last" if full_name.startswith(first_name) else "last_first",
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


if __name__ == "__main__":
    print(dict(os.environ))
    # asyncio.run(process(5, 0))
