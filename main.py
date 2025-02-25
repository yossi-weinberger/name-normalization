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

DATASET_PATH = "./lp_members.csv"
# DATASET_PATH = "./test_cases.csv"
RESULT_DIR_PATH = "./results/"
TASKS_PATH = "./tasks.jsonl"


RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "name_schema",
        "schema": {
            "type": "object",
            "required": ["is_person", "first_name", "last_name"],
            "properties": {
                "is_person": {
                    "type": "boolean",
                    "description": "Whether it is a person's name (not a business, institution, etc.).",
                },
                "last_name": {
                    "type": "string",
                    "description": "The last name",
                },
                "first_name": {
                    "type": "string",
                    "description": "The first name",
                },
            },
            "additionalProperties": False,
        },
        "strict": True,
    },
}

# PROMPT = (
#     "User messages are names to be split into given name and surname. "
#     "Name order may be either 'given family' and 'family given'. "
#     "Disregard any titles or text that is not a person's name. Keep the spelling as is."
# )

# PROMPT = (
#     "Analyze the given Hebrew text and determine: "
#     "1. Whether it is a person's name (not a business, institution, etc.). "
#     "2. If it is a person's name, identify the first name and last name and return them in the correct order (first-last). "
   
#     "Requirements: "
#     "1. The output must use exactly the same Hebrew characters as in the input. "
#     "2. No non-Hebrew characters should appear in the output. "
#     "3. The output should only include words that were present in the input."
# )

PROMPT = (
    "For Hebrew text: "
    "1. Is this a person (not business)? If no, return is_person=false "
    "2. If yes: extract first_name, last_name, ignoring titles (ד״ר, הרב, פרופ׳, עו״ד, etc.) "
    "3. Use only existing words, Hebrew chars only "
    "4. If both names could be first names or both could be family names, "
    "   keep the original order (first word as first_name, second as last_name) "
    "Return clean name without titles, exact substrings from input"
)



@throttle(rate_limit=500, period=60)
async def process_name(name: str):
    """
    Processes a single name using OpenAI's API to split it into first and last name.
    
    Args:
        name (str): The full name to process
        
    Returns:
        tuple: A tuple containing (first_name, last_name)
        
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
        response_format=RESPONSE_FORMAT,
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
    result_path = os.path.join(RESULT_DIR_PATH, f"{start:05d}-{(start + stop):05d}.csv")
    reject_path = os.path.join(RESULT_DIR_PATH, f"{start:05d}-{(start + stop):05d}_rejects.csv")
    
    result_file = open(result_path, "w", newline="", encoding="utf-8")
    reject_file = open(reject_path, "w", newline="", encoding="utf-8")
    
    result_writer = csv.DictWriter(result_file, fieldnames=["id", "first_name", "last_name"])
    reject_writer = csv.DictWriter(reject_file, fieldnames=["id", "name", "reason"])
    
    result_writer.writeheader()
    reject_writer.writeheader()
    
    return result_writer, reject_writer, result_file, reject_file

def validate_name_parts(full_name: str, first_name: str, last_name: str) -> tuple[bool, str]:
    """
    Validates that the output names only contain words from the input name.
    
    Args:
        full_name (str): The original full name
        first_name (str): The extracted first name
        last_name (str): The extracted last name
        
    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    input_words = set(full_name.split())
    output_words = set()
    
    if first_name:
        output_words.update(first_name.split())
    if last_name:
        output_words.update(last_name.split())
    
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
                parts["first_name"],
                parts["last_name"]
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
                "first_name": parts["first_name"],
                "last_name": parts["last_name"]
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
    writer, reject_writer, result_file, reject_file = setup_writers(start, stop)
    
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


asyncio.run(process(3, 0))


# Batch(id='batch_67b8ecac035c8190a4365678412c89ad', completion_window='24h', created_at=1740172460, endpoint='/v1/chat/completions', input_file_id='file-QUogCdMjGxSyVajahiPbGP', object='batch', status='validating', cancelled_at=None, cancelling_at=None, completed_at=None, error_file_id=None, errors=None, expired_at=None, expires_at=1740258860, failed_at=None, finalizing_at=None, in_progress_at=None, metadata=None, output_file_id=None, request_counts=BatchRequestCounts(completed=0, failed=0, total=0))
