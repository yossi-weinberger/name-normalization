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

load_dotenv()

DATASET_PATH = "./lp_members.csv"
RESULT_DIR_PATH = "./results/"
TASKS_PATH = "./tasks.jsonl"

client = OpenAI()
async_client = AsyncOpenAI()

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "name_schema",
        "schema": {
            "type": "object",
            "required": ["first_name", "last_name"],
            "properties": {
                "last_name": {
                    "type": "string",
                    "description": "The last name",
                },
                "first_name": {
                    "type": "string",
                    "description": "The first name",
                }
            },
            "additionalProperties": False,
        },
        "strict": True,
    },
}

PROMPT = (
    "User messages are names to be split into given name and surname. "
    "Name order may be either 'given family' and 'family given'. "
    "Disregard any titles or text that is not a person's name. Keep the spelling as is."
)


@throttle(rate_limit=500, period=60)
async def process_name(name: str):
    response = await async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": PROMPT,
            },
            {"role": "user", "content": name},
        ],
        response_format=RESPONSE_FORMAT,
        temperature=0.15,
        max_completion_tokens=128,
    )

    # הדפסת מידע על הטוקנים
    print(f"\033[36m[Tokens] {name}: completion={response.usage.completion_tokens}, prompt={response.usage.prompt_tokens}, total={response.usage.total_tokens}\033[0m")

    tracker.add_usage(response, model="gpt-4o-mini")

    parts = json.loads(response.choices[0].message.content)

    return (parts["first_name"], parts["last_name"])


# async def process_name_v2(id: str, name: str):
#     name_parts = name.split()

#     properties = {}
#     for i, part in enumerate(name_parts):
#         properties[f"part_{i}"] = {
#             "type": "string",
#             "description": f"Part {i + 1} of the name",
#             "enum": ["G", "F"],
#         }

#     response = (
#         await async_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": (
#                         PROMPT
#                         + " In the JSON output, each part of the name should be in its own field with a value of 'G' for given name and 'F' for family name."
#                     ),
#                 },
#                 {"role": "user", "content": name},
#             ],
#             response_format={
#                 "type": "json_schema",
#                 "json_schema": {
#                     "name": "name_schema",
#                     "schema": {
#                         "type": "object",
#                         "required": name_parts,
#                         "properties": properties,
#                         "additionalProperties": False,
#                     },
#                     "strict": True,
#                 },
#             },
#             temperature=0.15,
#             max_completion_tokens=128,
#         ),
#     )

#     parts = json.loads(response.choices[0].message.content)
#     first_name_parts = [part for part, value in parts.items() if value == "G"]
#     last_name_parts = [part for part, value in parts.items() if value == "F"]

#     first_name = name_parts
#     last_name = " ".join(last_name_parts)

#     return (id, first_name, last_name)

#     return (id,)


async def process(stop: int, start: int = 0) -> None:
    tracker.reset()

    result_path = os.path.join(RESULT_DIR_PATH, f"{start:05d}-{(start + stop):05d}lavi.csv")

    with (
        open(DATASET_PATH, "r", encoding="utf-8-sig") as src,
        open(result_path, "w", newline="", encoding="utf-8") as dst,
    ):
        reader = itertools.islice(csv.DictReader(src), start, stop)
        writer = csv.DictWriter(dst, fieldnames=["id", "first_name", "last_name"])
        writer.writeheader()

        chunk_size = 50
        while True:
            chunk = list(itertools.islice(reader, chunk_size))
            if not chunk:
                break

            chunk = filter(lambda r: is_name_valid(r["CUSTDES"]), chunk)

            tasks = [
                (
                    row["CUST"],
                    row["CUSTDES"],
                    process_name(row["CUSTDES"]),
                )
                for row in chunk
            ]
            responses = await asyncio.gather(*[task[2] for task in tasks])
            for (id, full_name, _), (first_name, last_name) in zip(tasks, responses):
                ai_name_parts = first_name.split() + last_name.split()
                src_name_parts = full_name.split()

                for part in ai_name_parts:
                    if part not in src_name_parts:
                        print(f"[{id}]AI part not found in source: {part}")

                writer.writerow(
                    {
                        "id": id,
                        "first_name": first_name,
                        "last_name": last_name,
                    }
                )

            dst.flush()

    tracker.print_summary()


def generate_tasks(top: int) -> None:
    with (
        open(DATASET_PATH, "r", encoding="utf-8-sig") as src,
        open(TASKS_PATH, "w") as dst,
    ):
        reader = itertools.islice(csv.DictReader(src), top)
        for row in reader:
            dst.write(
                json.dumps(
                    {
                        "custom_id": row["CUST"],
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "gpt-4o-mini",
                            "temperature": 0.1,
                            "max_completion_tokens": 256,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "User messages are names to be split into given name and surname. Name order may be either 'given family' and 'family given'. Disregard any titles.",
                                },
                                {"role": "user", "content": row["CUSTDES"]},
                            ],
                            "response_format": RESPONSE_FORMAT,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def create_batch():
    batch_file = client.files.create(
        file=open("./tasks.jsonl", "rb"),
        purpose="batch",
    )

    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    print(batch_job)


def is_name_valid(name: str) -> bool:
    return not re.search(r"[^א-תפףךםןץ\s']", name)


def print_names_with_special_characters():
    with open(DATASET_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        counter = 0
        for row in reader:
            if not is_name_valid(row["CUSTDES"]):
                print(row["CUSTDES"])
                counter += 1

        print(f"Total names with special characters: {counter}")


def get_batch(batch_id: str):
    batch_job = client.batches.retrieve(batch_id)

    print(batch_job)

    if batch_job.status == "completed":
        print("Batch is completed")
        content = client.files.content(batch_job.output_file_id).content
        with open("results.jsonl", "wb") as f:
            f.write(content)


# generate_tasks(1000)
# create_batch()
# get_batch("batch_67b8ecac035c8190a4365678412c89ad")
asyncio.run(process(3, 0))
# print_names_with_special_characters()


# Batch(id='batch_67b8ecac035c8190a4365678412c89ad', completion_window='24h', created_at=1740172460, endpoint='/v1/chat/completions', input_file_id='file-QUogCdMjGxSyVajahiPbGP', object='batch', status='validating', cancelled_at=None, cancelling_at=None, completed_at=None, error_file_id=None, errors=None, expired_at=None, expires_at=1740258860, failed_at=None, finalizing_at=None, in_progress_at=None, metadata=None, output_file_id=None, request_counts=BatchRequestCounts(completed=0, failed=0, total=0))
