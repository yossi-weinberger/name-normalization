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
client = OpenAI()
async_client = AsyncOpenAI()

# DATASET_PATH = "./lp_members.csv"
DATASET_PATH = "./test_cases.csv"
RESULT_DIR_PATH = "./results/"
TASKS_PATH = "./tasks.jsonl"


RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "name_schema",
        "schema": {
            "type": "object",
            "required": ["is_person", "given_name", "family_name"],
            "properties": {
                "is_person": {
                    "type": "boolean",
                    # "description": "Whether it is a person's name (not a business, institution, etc.).",
                },
                "family_name": {
                    "type": "string",
                    # "description": "The last name",
                },
                "given_name": {
                    "type": "string",
                    # "description": "The first name",
                },
            },
            "additionalProperties": False,
        },
        "strict": True,
    },
}

PROMPT = (
    "Your job is to normalize and split names into a JSON object with given and family names. "
    "User prompts are names to be normalized. Do the following:\n"
    "- Set is_person to true if the name is a person's name (not a business, institution, etc.), "
    "or contains an extractable person's name mixed with other text.\n"
    "- Extract given_name and family_name.\n"
    "- Drop titles (ד״ר, הרב, פרופ׳, עו״ד, etc.) from the output.\n"
    "- In case both names could be both given and family names, assume 'given family' order.\n"
    "- Keep the spelling as is.\n"
)   


def create_name_schema(name: str) -> dict:
    """Creates a schema that allows consecutive sequences from either start or end, excluding full-length sequences."""
    words = name.split()
    
    # Create sequences from both start and end, but not full length
    from_start = []
    from_end = []
    
    # Add sequences from start (excluding full length)
    for i in range(1, len(words)):  # Changed range to exclude full length
        from_start.append(" ".join(words[:i]))
    
    # Add sequences from end (excluding full length)
    for i in range(1, len(words)):  # Changed range to exclude full length
        from_end.append(" ".join(words[-i:]))
    
    # Both given_name and family_name can be from either direction
    name_options = from_start + from_end
    
    print(f"\nInput name: {name}")
    print("\nSequences from start:")
    for name in from_start:
        print(f"  {name}")
    print("\nSequences from end:")
    for name in from_end:
        print(f"  {name}")
    print("-" * 50)
    
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "name_schema",
            "schema": {
                "type": "object",
                "required": ["is_person", "given_name", "family_name"],
                "properties": {
                    "is_person": {
                        "type": "boolean",
                    },
                    "family_name": {
                        "type": "string",
                        "enum": name_options,
                    },
                    "given_name": {
                        "type": "string",
                        "enum": name_options,
                    },
                },
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

@throttle(rate_limit=500, period=60)
async def process_name(name: str):
    """
    Processes a single name using OpenAI's API to split it into first and last name.

    Args:
        name (str): The full name to process

    Returns:
        tuple: A tuple containing (given_name, family_name)

    Note:
        This function is throttled to 500 requests per 60 seconds
    """
    response = await async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": PROMPT,
            },
            {"role": "user", "content": name},
        ],
        # response_format=RESPONSE_FORMAT,
        response_format=create_name_schema(name),
        temperature=0.15,
        max_completion_tokens=128,
    )

    tracker.add_usage(response, model="gpt-4o-mini")

    return response


def setup_writers(start: int, stop: int):
    """
    Sets up CSV writers for results and rejections.

    Args:
        start (int): Starting index for file naming
        stop (int): Ending index for file naming

    Returns:
        tuple: (result_writer, reject_writer, result_file, reject_file)
    """
    result_path = os.path.join(
        RESULT_DIR_PATH, f"{start:05d}-{(start + stop):05d}.csv")
    reject_path = os.path.join(
        RESULT_DIR_PATH, f"{start:05d}-{(start + stop):05d}_rejects.csv")

    result_file = open(result_path, "w", newline="", encoding="utf-8")
    reject_file = open(reject_path, "w", newline="", encoding="utf-8")

    result_writer = csv.DictWriter(result_file, fieldnames=[
                                   "id", "given_name", "family_name"])
    reject_writer = csv.DictWriter(reject_file, fieldnames=[
                                   "id", "name", "reason"])

    result_writer.writeheader()
    reject_writer.writeheader()

    return result_writer, reject_writer, result_file, reject_file


def validate_name_parts(full_name: str, given_name: str, family_name: str) -> tuple[bool, str]:
    """
    Validates that the output names only contain words from the input name.

    Args:
        full_name (str): The original full name
        given_name (str): The extracted first name
        family_name (str): The extracted last name

    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    input_words = set(full_name.split())
    output_words = set()

    if given_name:
        output_words.update(given_name.split())
    if family_name:
        output_words.update(family_name.split())

    invalid_words = output_words - input_words
    if invalid_words:
        return False, f"validation_error: output contains invalid words: {', '.join(invalid_words)}"

    return True, ""


async def process_chunk(chunk: list, writer, reject_writer):
    """
    Processes a chunk of names using OpenAI API.

    Args:
        chunk (list): List of names to process
        writer: CSV writer for valid results
        reject_writer: CSV writer for rejected entries
    """
    tasks = []

    for row in chunk:
        if not is_name_valid(row["CUSTDES"]):
            # Write invalid names to reject file
            reject_writer.writerow({
                "id": row["CUST"],
                "name": row["CUSTDES"],
                "reason": "invalid_characters"
            })
        else:
            # Add valid names to processing tasks
            tasks.append((
                row["CUST"],
                row["CUSTDES"],
                process_name(row["CUSTDES"])
            ))

    # Process valid names with API
    responses = await asyncio.gather(*[task[2] for task in tasks])

    # Handle API responses
    for (id, full_name), response in zip([(t[0], t[1]) for t in tasks], responses):
        try:
            parts = json.loads(response.choices[0].message.content)

            if not parts["is_person"]:
                # Write to rejects if not a person
                reject_writer.writerow({
                    "id": id,
                    "name": full_name,
                    "reason": "not_a_person"
                })
                continue

            # Validate name parts using validation function
            is_valid, error_message = validate_name_parts(
                full_name,
                parts["given_name"],
                parts["family_name"]
            )

            if not is_valid:
                reject_writer.writerow({
                    "id": id,
                    "name": full_name,
                    "reason": error_message
                })
                continue

            # Write valid person names to results
            writer.writerow({
                "id": id,
                "given_name": parts["given_name"],
                "family_name": parts["family_name"]
            })
        except Exception as e:
            # Write processing errors to rejects
            reject_writer.writerow({
                "id": id,
                "name": full_name,
                "reason": f"processing_error: {str(e)}"
            })


async def process(stop: int, start: int = 0) -> None:
    """
    Main processing function that handles name processing pipeline.

    Args:
        stop (int): Number of records to process
        start (int): Starting index (default: 0)

    Process:
        1. Sets up CSV writers
        2. Reads names in chunks
        3. Processes each chunk
        4. Writes results to appropriate files
    """
    writer, reject_writer, result_file, reject_file = setup_writers(
        start, stop)

    with open(DATASET_PATH, "r", encoding="utf-8-sig") as src:
        reader = itertools.islice(csv.DictReader(src), start, stop)

        while True:
            # Process in chunks of 50
            chunk = list(itertools.islice(reader, 50))
            if not chunk:
                break

            await process_chunk(chunk, writer, reject_writer)

            # Ensure data is written to files
            result_file.flush()
            reject_file.flush()

    # Clean up
    result_file.close()
    reject_file.close()

    # print token usage summary
    tracker.print_summary()


def is_name_valid(name: str) -> bool:
    """
    Validates if a name contains only Hebrew characters, spaces, and apostrophes.

    Args:
        name (str): The name to validate

    Returns:
        bool: True if the name contains only valid characters, False otherwise
    """
    return not re.search(r"[^א-תפףךםןץ\s']", name)


def generate_tasks(top: int) -> None:
    """
    Generates a JSONL file containing OpenAI API tasks for batch processing.

    Args:
        top (int): Number of names to process from the dataset

    Creates a JSONL file with API requests formatted for OpenAI's batch processing API.
    Each request includes the name to process and necessary API parameters.
    """
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
                            "temperature": 0.15,
                            "max_completion_tokens": 128,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": PROMPT,
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
    """
    Creates a new batch processing job with OpenAI's API.

    1. Uploads the tasks.jsonl file to OpenAI
    2. Creates a new batch processing job
    3. Prints the batch job details
    """
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


def get_batch(batch_id: str):
    """
    Retrieves and processes the results of a batch job.

    Args:
        batch_id (str): The ID of the batch job to retrieve

    If the batch is completed:
        1. Downloads the results
        2. Saves them to results.jsonl
    """
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

asyncio.run(process(77, 0))

# Batch(id='batch_67b8ecac035c8190a4365678412c89ad', completion_window='24h', created_at=1740172460, endpoint='/v1/chat/completions', input_file_id='file-QUogCdMjGxSyVajahiPbGP', object='batch', status='validating', cancelled_at=None, cancelling_at=None, completed_at=None, error_file_id=None, errors=None, expired_at=None, expires_at=1740258860, failed_at=None, finalizing_at=None, in_progress_at=None, metadata=None, output_file_id=None, request_counts=BatchRequestCounts(completed=0, failed=0, total=0))
