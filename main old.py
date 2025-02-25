# from openai import OpenAI, AsyncOpenAI
# from dotenv import load_dotenv
# from throttler import throttle
# import asyncio
# import csv
# import itertools
# import json
# import os
# import re
# from usage_tracker import tracker

# load_dotenv()

# DATASET_PATH = "./lp_members.csv"
# # DATASET_PATH = "./test_cases.csv"
# RESULT_DIR_PATH = "./results/"
# TASKS_PATH = "./tasks.jsonl"

# client = OpenAI()
# async_client = AsyncOpenAI()

# """
# מערכת להפרדת שמות עבריים לשם פרטי ושם משפחה
# משתמש ב-OpenAI API כדי לנתח שמות מקובץ CSV ולייצר פלט מאורגן

# קבצי קלט/פלט:
# - קובץ קלט: CSV עם שמות לעיבוד
# - קובץ פלט: CSV עם השמות המופרדים
# - קובץ משימות: JSONL עם בקשות API מוכנות
# """

# # קבועים לתבנית התשובה מ-OpenAI API
# RESPONSE_FORMAT = {
#     "type": "json_schema",
#     "json_schema": {
#         "name": "name_schema",
#         "schema": {
#             "type": "object",
#             "required": [
#                 "is_person",
#                 "first_name",
#                 "last_name",
#                 "confidence",
#                 "name_order",
#             ],
#             "properties": {
#                 "is_person": {
#                     "type": "boolean",
#                     "description": "האם זה שם של אדם",
#                 },
#                 "first_name": {
#                     "type": "string",
#                     "description": "השם הפרטי",
#                 },
#                 "last_name": {
#                     "type": "string",
#                     "description": "שם המשפחה",
#                 },
#                 "confidence": {
#                     "type": "number",
#                     "description": "רמת הביטחון בניתוח (0-1)",
#                 },
#                 "name_order": {
#                     "type": "string",
#                     "enum": ["first_last", "last_first"],
#                     "description": "סדר השמות במקור",
#                 },
#             },
#             "additionalProperties": False,
#         },
#         "strict": True,
#     },
# }

# PROMPT = """
# Analyze the given Hebrew text and determine:
# 1. Is this a person's name (not a business, institution, etc.)
# 2. What is the first name and last name
# 3. Is the name in regular order (first-last) or reversed (last-first)

# Examples:
# - "רהיטי הירדן" -> not a person's name
# - "יוסי כהן" -> person's name, regular order
# - "רוזנבלט יוסי" -> person's name, reversed order

# Maintain the original spelling and do not modify it.
# """

# # יצירת אובייקט Throttler
# throttler = throttle(rate_limit=500, period=60)

# @throttler
# async def process_name(name: str) -> tuple[str, str]:
#     """
#     מעבד שם בודד באמצעות OpenAI API
    
#     Args:
#         name: השם המלא לעיבוד
        
#     Returns:
#         tuple המכיל את השם הפרטי ושם המשפחה
#     """
#     try:
#         response = await async_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": PROMPT},
#                 {"role": "user", "content": name}
#             ],
#             response_format=RESPONSE_FORMAT
#         )
        
#         # מעקב אחר שימוש
#         tracker.add_usage(response, "gpt-4")
        
#         result = json.loads(response.choices[0].message.content)
#         if result["is_person"]:
#             return (result["first_name"], result["last_name"])
#         else:
#             return (None, None)
#     except Exception as e:
#         print(f"Error processing name: {name}, {e}")
#         return (None, None)


# # async def process_name_v2(id: str, name: str):
# #     name_parts = name.split()

# #     properties = {}
# #     for i, part in enumerate(name_parts):
# #         properties[f"part_{i}"] = {
# #             "type": "string",
# #             "description": f"Part {i + 1} of the name",
# #             "enum": ["G", "F"],
# #         }

# #     response = (
# #         await async_client.chat.completions.create(
# #             model="gpt-4o-mini",
# #             messages=[
# #                 {
# #                     "role": "system",
# #                     "content": (
# #                         PROMPT
# #                         + " In the JSON output, each part of the name should be in its own field with a value of 'G' for given name and 'F' for family name."
# #                     ),
# #                 },
# #                 {"role": "user", "content": name},
# #             ],
# #             response_format={
# #                 "type": "json_schema",
# #                 "json_schema": {
# #                     "name": "name_schema",
# #                     "schema": {
# #                         "type": "object",
# #                         "properties": properties,
# #                         "additionalProperties": False,    "required": name_parts,
# #                     
# #                     },
# #                     "strict": True,
# #                 },
# #             },
# #             temperature=0.15,
# #             max_completion_tokens=128,
# #         ),
# #     )

# #     parts = json.loads(response.choices[0].message.content)
# #     first_name_parts = [part for part, value in parts.items() if value == "G"]
# #     last_name_parts = [part for part, value in parts.items() if value == "F"]

# #     first_name = name_parts
# #     last_name = " ".join(last_name_parts)

# #     return (id, first_name, last_name)

# #     return (id,)


# async def process(stop: int, start: int = 0) -> None:
#     """
#     מעבד קבוצת שמות מתוך קובץ הקלט

#     Args:
#         stop: אינדקס הסיום לעיבוד
#         start: אינדקס ההתחלה לעיבוד (ברירת מחדל: 0)
#     """
#     result_path = os.path.join(RESULT_DIR_PATH, f"{start:05d}-{(start + stop):05d}.csv")
#     reject_path = os.path.join(
#         RESULT_DIR_PATH, f"{start:05d}-{(start + stop):05d}_rejects.csv"
#     )

#     with (
#         open(DATASET_PATH, "r", encoding="utf-8-sig") as src,
#         open(result_path, "w", newline="", encoding="utf-8") as dst,
#         open(reject_path, "w", newline="", encoding="utf-8") as reject_dst,
#     ):
#         reader = itertools.islice(csv.DictReader(src), start, stop)
#         writer = csv.DictWriter(
#             dst, fieldnames=["id", "first_name", "last_name", "original_order"]
#         )
#         reject_writer = csv.DictWriter(reject_dst, fieldnames=["id", "name", "reason"])

#         writer.writeheader()
#         reject_writer.writeheader()

#         chunk_size = 50
#         while True:
#             chunk = list(itertools.islice(reader, chunk_size))
#             if not chunk:
#                 break

#             chunk = filter(lambda r: is_name_valid(r["CUSTDES"]), chunk)

#             tasks = [
#                 (row["CUST"], row["CUSTDES"], process_name(row["CUSTDES"]))
#                 for row in chunk
#             ]
            
#             responses = await asyncio.gather(*[task[2] for task in tasks])
            
#             for (id, full_name), response in zip([(t[0], t[1]) for t in tasks], responses):
#                 if response[0] is None and response[1] is None:
#                     reject_writer.writerow({
#                         "id": id,
#                         "name": full_name,
#                         "reason": "not_a_person"
#                     })
#                     continue

#                 writer.writerow({
#                     "id": id,
#                     "first_name": response[0],
#                     "last_name": response[1],
#                     "original_order": "first_last" if response[0] and response[1] else "last_first",
#                 })

#             dst.flush()
#             reject_dst.flush()

#     # הדפסת סיכום בסוף העיבוד
#     tracker.print_summary()


# def generate_tasks(top: int) -> None:
#     """
#     מייצר קובץ JSONL עם משימות לעיבוד באצווה

#     Args:
#         top: מספר השורות לעיבוד מתחילת הקובץ
#     """
#     with (
#         open(DATASET_PATH, "r", encoding="utf-8-sig") as src,
#         open(TASKS_PATH, "w") as dst,
#     ):
#         reader = itertools.islice(csv.DictReader(src), top)
#         for row in reader:
#             dst.write(
#                 json.dumps(
#                     {
#                         "custom_id": row["CUST"],
#                         "method": "POST",
#                         "url": "/v1/chat/completions",
#                         "body": {
#                             "model": "gpt-4o-mini",
#                             "temperature": 0.1,
#                             "max_completion_tokens": 256,
#                             "messages": [
#                                 {
#                                     "role": "system",
#                                     "content": "User messages are names to be split into given name and surname. Name order may be either 'given family' and 'family given'. Disregard any titles.",
#                                 },
#                                 {"role": "user", "content": row["CUSTDES"]},
#                             ],
#                             "response_format": RESPONSE_FORMAT,
#                         },
#                     },
#                     ensure_ascii=False,
#                 )
#                 + "\n"
#             )


# def create_batch():
#     batch_file = client.files.create(
#         file=open("./tasks.jsonl", "rb"),
#         purpose="batch",
#     )

#     batch_job = client.batches.create(
#         input_file_id=batch_file.id,
#         endpoint="/v1/chat/completions",
#         completion_window="24h",
#     )

#     print(batch_job)


# def is_name_valid(name: str) -> bool:
#     """
#     בודק אם השם מכיל רק תווים עבריים חוקיים

#     Args:
#         name: השם לבדיקה

#     Returns:
#         True אם השם תקין, False אחרת
#     """
#     return not re.search(r"[^א-תפףךםןץ\s']", name)


# def print_names_with_special_characters():
#     with open(DATASET_PATH, "r", encoding="utf-8-sig") as f:
#         reader = csv.DictReader(f)

#         counter = 0
#         for row in reader:
#             if not is_name_valid(row["CUSTDES"]):
#                 print(row["CUSTDES"])
#                 counter += 1

#         print(f"Total names with special characters: {counter}")


# def get_batch(batch_id: str):
#     batch_job = client.batches.retrieve(batch_id)

#     print(batch_job)

#     if batch_job.status == "completed":
#         print("Batch is completed")
#         content = client.files.content(batch_job.output_file_id).content
#         with open("results.jsonl", "wb") as f:
#             f.write(content)


# # generate_tasks(1000)
# # create_batch()
# # get_batch("batch_67b8ecac035c8190a4365678412c89ad")
# asyncio.run(process(70, 0))
# # print_names_with_special_characters()


# # Batch(id='batch_67b8ecac035c8190a4365678412c89ad', completion_window='24h', created_at=1740172460, endpoint='/v1/chat/completions', input_file_id='file-QUogCdMjGxSyVajahiPbGP', object='batch', status='validating', cancelled_at=None, cancelling_at=None, completed_at=None, error_file_id=None, errors=None, expired_at=None, expires_at=1740258860, failed_at=None, finalizing_at=None, in_progress_at=None, metadata=None, output_file_id=None, request_counts=BatchRequestCounts(completed=0, failed=0, total=0))
