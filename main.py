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
import glob
import time

load_dotenv()
client = OpenAI()
async_client = AsyncOpenAI()

DATASET_PATH = "./lp_members.csv"
# DATASET_PATH = "./test_cases.csv"
RESULT_DIR_PATH = "./results/"
TASKS_DIR = "./tasks/"
TASKS_PATH = os.path.join(TASKS_DIR, "tasks.jsonl")
BATCH_TRACKING_FILE = "batch_tracking.txt"


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

# region: old code
# @throttle(rate_limit=500, period=60)
# async def process_name(name: str):
#     """
#     Processes a single name using OpenAI's API to split it into first and last name.

#     Args:
#         name (str): The full name to process

#     Returns:
#         tuple: A tuple containing (given_name, family_name)

#     Note:
#         This function is throttled to 500 requests per 60 seconds
#     """
#     response = await async_client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {
#                 "role": "system",
#                 "content": PROMPT,
#             },
#             {"role": "user", "content": name},
#         ],
#         # response_format=RESPONSE_FORMAT,
#         response_format=create_name_schema(name),
#         temperature=0.15,
#         max_completion_tokens=128,
#     )

#     tracker.add_usage(response, model="gpt-4o-mini")

#     return response

# def setup_writers(start: int, stop: int):
#     """
#     Sets up CSV writers for results and rejections.

#     Args:
#         start (int): Starting index for file naming
#         stop (int): Ending index for file naming

#     Returns:
#         tuple: (result_writer, reject_writer, result_file, reject_file)
#     """
#     result_path = os.path.join(
#         RESULT_DIR_PATH, f"{start:05d}-{(start + stop):05d}.csv")
#     reject_path = os.path.join(
#         RESULT_DIR_PATH, f"{start:05d}-{(start + stop):05d}_rejects.csv")

#     result_file = open(result_path, "w", newline="", encoding="utf-8")
#     reject_file = open(reject_path, "w", newline="", encoding="utf-8")

#     result_writer = csv.DictWriter(result_file, fieldnames=[
#                                    "id", "given_name", "family_name"])
#     reject_writer = csv.DictWriter(reject_file, fieldnames=[
#                                    "id", "name", "reason"])

#     result_writer.writeheader()
#     reject_writer.writeheader()

#     return result_writer, reject_writer, result_file, reject_file

# def validate_name_parts(full_name: str, given_name: str, family_name: str) -> tuple[bool, str]:
#     """
#     Validates that the output names only contain words from the input name.

#     Args:
#         full_name (str): The original full name
#         given_name (str): The extracted first name
#         family_name (str): The extracted last name

#     Returns:
#         tuple[bool, str]: (is_valid, error_message)
#     """
#     input_words = set(full_name.split())
#     output_words = set()

#     if given_name:
#         output_words.update(given_name.split())
#     if family_name:
#         output_words.update(family_name.split())

#     invalid_words = output_words - input_words
#     if invalid_words:
#         return False, f"validation_error: output contains invalid words: {', '.join(invalid_words)}"

#     return True, ""

# async def process_chunk(chunk: list, writer, reject_writer):
#     """
#     Processes a chunk of names using OpenAI API.

#     Args:
#         chunk (list): List of names to process
#         writer: CSV writer for valid results
#         reject_writer: CSV writer for rejected entries
#     """
#     tasks = []

#     for row in chunk:
#         if not is_name_valid(row["CUSTDES"]):
#             # Write invalid names to reject file
#             reject_writer.writerow({
#                 "id": row["CUST"],
#                 "name": row["CUSTDES"],
#                 "reason": "invalid_characters"
#             })
#         else:
#             # Add valid names to processing tasks
#             tasks.append((
#                 row["CUST"],
#                 row["CUSTDES"],
#                 process_name(row["CUSTDES"])
#             ))

#     # Process valid names with API
#     responses = await asyncio.gather(*[task[2] for task in tasks])

#     # Handle API responses
#     for (id, full_name), response in zip([(t[0], t[1]) for t in tasks], responses):
#         try:
#             parts = json.loads(response.choices[0].message.content)

#             if not parts["is_person"]:
#                 # Write to rejects if not a person
#                 reject_writer.writerow({
#                     "id": id,
#                     "name": full_name,
#                     "reason": "not_a_person"
#                 })
#                 continue

#             # Validate name parts using validation function
#             is_valid, error_message = validate_name_parts(
#                 full_name,
#                 parts["given_name"],
#                 parts["family_name"]
#             )

#             if not is_valid:
#                 reject_writer.writerow({
#                     "id": id,
#                     "name": full_name,
#                     "reason": error_message
#                 })
#                 continue

#             # Write valid person names to results
#             writer.writerow({
#                 "id": id,
#                 "given_name": parts["given_name"],
#                 "family_name": parts["family_name"]
#             })
#         except Exception as e:
#             # Write processing errors to rejects
#             reject_writer.writerow({
#                 "id": id,
#                 "name": full_name,
#                 "reason": f"processing_error: {str(e)}"
#             })

# async def process(stop: int, start: int = 0) -> None:
#     """
#     Main processing function that handles name processing pipeline.

#     Args:
#         stop (int): Number of records to process
#         start (int): Starting index (default: 0)

#     Process:
#         1. Sets up CSV writers
#         2. Reads names in chunks
#         3. Processes each chunk
#         4. Writes results to appropriate files
#     """
#     writer, reject_writer, result_file, reject_file = setup_writers(
#         start, stop)

#     with open(DATASET_PATH, "r", encoding="utf-8-sig") as src:
#         reader = itertools.islice(csv.DictReader(src), start, stop)

#         while True:
#             # Process in chunks of 50
#             chunk = list(itertools.islice(reader, 50))
#             if not chunk:
#                 break

#             await process_chunk(chunk, writer, reject_writer)

#             # Ensure data is written to files
#             result_file.flush()
#             reject_file.flush()

#     # Clean up
#     result_file.close()
#     reject_file.close()

#     # print token usage summary
#     tracker.print_summary()

# endregion: old code
def is_name_valid(name: str) -> bool:
    """
    Validates if a name contains only Hebrew characters, spaces, and apostrophes.

    Args:
        name (str): The name to validate

    Returns:
        bool: True if the name contains only valid characters, False otherwise
    """
    return not re.search(r"[^א-תפףךםןץ\s']", name)


def create_name_schema(name: str) -> dict:
    """Creates a schema that allows consecutive sequences from either start or end, excluding full-length sequences."""
    words = name.split()
    
    # Create sequences from both start and end, but not full length
    from_start = []
    from_end = []
    
    # Add sequences from start (excluding full length)
    for i in range(1, len(words)):  # Changed range to exclude full length
        from_start.append("_".join(words[:i]))
    
    # Add sequences from end (excluding full length)
    for i in range(1, len(words)):  # Changed range to exclude full length
        from_end.append("_".join(words[-i:]))
    
    # Both given_name and family_name can be from either direction
    name_options = from_start + from_end
    
    # print(f"\nInput name: {name}")
    # print("\nSequences from start:")
    # for name in from_start:
    #     print(f"  {name}")
    # print("\nSequences from end:")
    # for name in from_end:
    #     print(f"  {name}")
    # print("-" * 50)
    
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

def generate_tasks(stop: int, start: int = 0) -> str:
    """
    Generates a JSONL file containing OpenAI API tasks for batch processing.

    Args:
        stop (int): Number of records to process
        start (int): Starting index (default: 0)

    Returns:
        str: Path to the generated tasks file

    Creates a JSONL file with API requests formatted for OpenAI's batch processing API.
    Each request includes the name to process and necessary API parameters.
    """
    # Create tasks directory if it doesn't exist
    if not os.path.exists(TASKS_DIR):
        os.makedirs(TASKS_DIR)
    
    # Create batch_results directory if it doesn't exist
    if not os.path.exists("batch_results"):
        os.makedirs("batch_results")
        
    # Generate filenames
    tasks_path = os.path.join(TASKS_DIR, f"tasks_{start:05d}-{stop:05d}.jsonl")
    range_str = f"{start:05d}-{stop:05d}"
    reject_path = os.path.join("batch_results", f"results_{range_str}_rejects.csv")
    
    with (
        open(DATASET_PATH, "r", encoding="utf-8-sig") as src,
        open(tasks_path, "w") as dst,
        open(reject_path, "a", newline="", encoding="utf-8") as reject_file,  # Changed to append mode
    ):
        reader = itertools.islice(csv.DictReader(src), start, stop)
        
        # Check if file is empty before writing header
        reject_file.seek(0, 2)  # Go to end of file
        is_empty = reject_file.tell() == 0
        reject_writer = csv.DictWriter(reject_file, fieldnames=["id", "name", "reason"])
        if is_empty:
            reject_writer.writeheader()
        
        for row in reader:
            name = row["CUSTDES"]
            if not is_name_valid(name):
                reject_writer.writerow({
                    "id": row["CUST"],
                    "name": name,
                    "reason": "invalid_characters"
                })
                continue
                
            # response_format = create_name_schema(name)
            dst.write(
                json.dumps(
                    {
                        "custom_id": row["CUST"],
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "gpt-4o-mini",
                            "temperature": 0.15,
                            "max_completion_tokens": 40,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": PROMPT,
                                },
                                {"role": "user", "content": name},
                            ],
                            "response_format": RESPONSE_FORMAT,
                            # "response_format": response_format,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    
    return tasks_path

def create_batch(stop: int, start: int = 0):
    """
    Creates a new batch processing job with OpenAI's API.

    Args:
        stop (int): Number of records to process
        start (int): Starting index (default: 0)

    1. Finds the specific tasks file for the given range
    2. Uploads the tasks.jsonl file to OpenAI
    3. Creates a new batch processing job
    4. Adds the batch to tracking file
    """
    # Construct the expected filename for this range
    tasks_file_path = os.path.join(TASKS_DIR, f"tasks_{start:05d}-{stop:05d}.jsonl")
    
    if not os.path.exists(tasks_file_path):
        raise FileNotFoundError(f"Tasks file not found for range {start}-{stop}. Please generate it first using generate_tasks({stop}, {start})")
    
    batch_file = client.files.create(
        file=open(tasks_file_path, "rb"),
        purpose="batch",
    )

    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    
    # Add to tracking file
    add_batch_to_tracking(batch_job.id, start, stop, tasks_file_path)
    
    print(f"Created new batch job: {batch_job.id}")

def format_batch_status(batch_job) -> str:
    """
    Formats batch job status in a readable way, including token usage and cost from the tracker.
    
    Args:
        batch_job: The batch job object
        
    Returns:
        str: Formatted status string
    """
    # Get usage data from tracker
    usage_data = tracker.get_summary()
    
    status_lines = [
        f"Task ID: {batch_job.id}",
        f"Status: {batch_job.status}",
        f"Progress: {batch_job.request_counts.completed}/{batch_job.request_counts.total} completed",
        f"Failed: {batch_job.request_counts.failed}",
        "",
        f"API Calls: {usage_data['total_calls']}",
        f"Total Tokens: {usage_data['total_tokens']:,}",
        f"Total Cost: ${usage_data['total_cost']:.4f}",
    ]
    return "\n".join(status_lines)

def add_batch_to_tracking(batch_id: str, start: int, stop: int, tasks_file: str):
    """
    Adds a batch job to the tracking file.
    
    Args:
        batch_id (str): The ID of the batch job
        start (int): Start of the range
        stop (int): End of the range
        tasks_file (str): Path to the tasks file
    """
    with open(BATCH_TRACKING_FILE, "a", encoding="utf-8") as f:
        f.write(f"{start:05d}-{stop:05d}|{batch_id}|{tasks_file}|{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

def get_batch_id_by_range(start: int, stop: int) -> str:
    """
    Finds the most recent batch ID for a given range from the tracking file.
    
    Args:
        start (int): Start of the range
        stop (int): End of the range
        
    Returns:
        str: Most recent batch ID if found, None otherwise
    """
    range_str = f"{start:05d}-{stop:05d}"
    if not os.path.exists(BATCH_TRACKING_FILE):
        return None
        
    latest_batch_id = None
    with open(BATCH_TRACKING_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                range_part, batch_id, *_ = line.strip().split("|")
                if range_part == range_str:
                    latest_batch_id = batch_id
    return latest_batch_id

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

def process_batch_results(content: bytes) -> tuple[list, list]:
    """
    Process the batch results to extract both successful results and rejections.
    
    Args:
        content (bytes): Raw content from the API
        
    Returns:
        tuple: (successful_results, rejected_results)
    """
    successful_results = []
    rejected_results = []
    
    # Load original data for validation
    original_data = {}
    with open(DATASET_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_data[row["CUST"]] = row["CUSTDES"]
    
    for line in content.decode('utf-8').splitlines():
        if not line.strip():
            continue
            
        data = json.loads(line)
        custom_id = data.get('custom_id', 'unknown')
        original_name = original_data.get(custom_id, "unknown")
        
        try:
            # Check if there's an error
            if data.get('error'):
                rejected_results.append({
                    'id': custom_id,
                    'name': original_name,
                    'reason': f'api_error: {data["error"]}'
                })
                continue
                
            # Get response content
            response_content = data['response']['body']['choices'][0]['message']['content']
            name_data = json.loads(response_content)
            
            if name_data['is_person']:
                # Validate output against input
                given_name = name_data['given_name'].replace('_', ' ')
                family_name = name_data['family_name'].replace('_', ' ')
                
                is_valid, error_message = validate_name_parts(original_name, given_name, family_name)
                
                if is_valid:
                    successful_results.append({
                        'id': custom_id,
                        'given_name': given_name,
                        'family_name': family_name
                    })
                else:
                    rejected_results.append({
                        'id': custom_id,
                        'name': original_name,
                        'reason': error_message
                    })
            else:
                rejected_results.append({
                    'id': custom_id,
                    'name': original_name,
                    'reason': 'not_a_person'
                })
                
        except Exception as e:
            rejected_results.append({
                'id': custom_id,
                'name': original_name,
                'reason': f'processing_error: {str(e)}'
            })
    
    return successful_results, rejected_results

def get_batch(stop: int, start: int = 0):
    """
    Retrieves and processes the results of a batch job for a specific range.

    Args:
        stop (int): End of the range
        start (int): Start of the range (default: 0)
    """
    batch_id = get_batch_id_by_range(start, stop)
    if not batch_id:
        print(f"No batch job found for range {start:05d}-{stop:05d}")
        return
        
    batch_job = client.batches.retrieve(batch_id)
    
    # Create batch_results directory if it doesn't exist
    if not os.path.exists("batch_results"):
        os.makedirs("batch_results")
    
    # Reset tracker
    tracker.reset()
    
    # If batch is completed, calculate usage from results
    if batch_job.status == "completed":
        content = client.files.content(batch_job.output_file_id).content
        for line in content.decode('utf-8').splitlines():
            if not line.strip():
                continue
                
            data = json.loads(line)
            if 'response' in data and 'body' in data['response'] and 'usage' in data['response']['body']:
                usage = data['response']['body']['usage']
                class MockResponse:
                    def __init__(self, usage_data):
                        self.usage = type('Usage', (), {
                            'prompt_tokens': usage_data.get('prompt_tokens', 0),
                            'completion_tokens': usage_data.get('completion_tokens', 0),
                            'total_tokens': usage_data.get('total_tokens', 0)
                        })
                
                mock_response = MockResponse(usage)
                tracker.add_usage(mock_response, model="gpt-4o-mini")
    
    # Get status string
    status_str = format_batch_status(batch_job)
    
    # Write detailed status to log file
    range_str = f"{start:05d}-{stop:05d}"
    log_path = os.path.join("batch_results", f"batch_{range_str}_status_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
        f.write("=" * 50 + "\n")
        f.write(f"Task ID: {batch_job.id}\n")
        f.write(f"Status: {batch_job.status}\n")
        f.write(f"Progress: {batch_job.request_counts.completed}/{batch_job.request_counts.total} completed\n")
        f.write(f"Failed: {batch_job.request_counts.failed}\n")
        f.write(f"Created at: {batch_job.created_at}\n")
        f.write(f"Expires at: {batch_job.expires_at}\n\n")
        usage_data = tracker.get_summary()
        f.write("Usage Information:\n")
        f.write(f"API Calls: {usage_data['total_calls']}\n")
        f.write(f"Input Tokens: {usage_data['input_tokens']:,}\n")
        f.write(f"Output Tokens: {usage_data['output_tokens']:,}\n")
        f.write(f"Total Tokens: {usage_data['total_tokens']:,}\n")
        f.write(f"Total Cost: ${usage_data['total_cost']:.4f}\n")
        f.write("=" * 50 + "\n\n")
    
    # Print minimal status to console
    print(status_str)

    if batch_job.status == "completed":
        result_path = os.path.join("batch_results", f"results_{range_str}.csv")
        reject_path = os.path.join("batch_results", f"results_{range_str}_rejects.csv")
        
        result_file = open(result_path, "w", newline="", encoding="utf-8")
        reject_file = open(reject_path, "a", newline="", encoding="utf-8")
        
        result_writer = csv.DictWriter(result_file, fieldnames=["id", "given_name", "family_name"])
        reject_writer = csv.DictWriter(reject_file, fieldnames=["id", "name", "reason"])
        
        # Write headers only if file is empty
        result_writer.writeheader()
        reject_file.seek(0, 2)  # Go to end of file
        if reject_file.tell() == 0:  # If file is empty
            reject_writer.writeheader()
            
        # Get and process results
        content = client.files.content(batch_job.output_file_id).content
        successful_results, rejected_results = process_batch_results(content)
        
        # Write results using the CSV writers
        for result in successful_results:
            result_writer.writerow(result)
            
        for reject in rejected_results:
            reject_writer.writerow(reject)
            
        # Close the files
        result_file.close()
        reject_file.close()
            
        print(f"\nResults Summary:")
        print(f"✓ Successful: {len(successful_results)}")
        print(f"✗ Rejected: {len(rejected_results)}")
        print(f"\nFiles saved to:")
        print(f"- {result_path}")
        print(f"- {reject_path}")

# generate_tasks(171, 0)
# create_batch(170, 0)
# get_batch(171, 0)

# get_batch("batch_67b8ecac035c8190a4365678412c89ad")

# asyncio.run(process(77, 0))

# Batch(id='batch_67b8ecac035c8190a4365678412c89ad', completion_window='24h', created_at=1740172460, endpoint='/v1/chat/completions', input_file_id='file-QUogCdMjGxSyVajahiPbGP', object='batch', status='validating', cancelled_at=None, cancelling_at=None, completed_at=None, error_file_id=None, errors=None, expired_at=None, expires_at=1740258860, failed_at=None, finalizing_at=None, in_progress_at=None, metadata=None, output_file_id=None, request_counts=BatchRequestCounts(completed=0, failed=0, total=0))


# # קבוצה 1: 0-10000
# generate_tasks(10000, 0)
# # קבוצה 2: 10000-20000
# generate_tasks(20000, 10000)
# # קבוצה 3: 20000-30000
# generate_tasks(30000, 20000)
# # קבוצה 4: 30000-40000
# generate_tasks(40000, 30000)
# # קבוצה 5: 40000-50000
# generate_tasks(50000, 40000)
# # קבוצה 6: 50000-60000
# generate_tasks(60000, 50000)
# # קבוצה 7: 60000-70000
# generate_tasks(70000, 60000)
# כל המשימות נוצרו!!!



# קבוצה 1 נשלח!!!
# create_batch(10000, 0)
# קבוצה 2 נשלח!!!
# create_batch(20000, 10000)
# # קבוצה 3 נשלח!!!
# create_batch(30000, 20000)
# # קבוצה 4 נשלח!!!
# create_batch(40000, 30000)
# # קבוצה 5 נשלח!!!
# create_batch(50000, 40000)
# # קבוצה 6 נשלח!!!
# create_batch(60000, 50000)
# # קבוצה 7 נשלח!!!
# create_batch(70000, 60000)




# # קבוצה 1 נמשך לקבצים!!!!
# get_batch(10000, 0) 
# # קבוצה 2 נמשך לקבצים!!!!
# get_batch(20000, 10000)
# # קבוצה 3 נמשך לקבצים!!!!
# get_batch(30000, 20000)
# # קבוצה 4 נמשך לקבצים!!!!
# get_batch(40000, 30000)
# # קבוצה 5 נמשך לקבצים!!!!
# get_batch(50000, 40000)
# # קבוצה 6 נמשך לקבצים!!!!
# get_batch(60000, 50000)
# # קבוצה 7
# get_batch(70000, 60000)