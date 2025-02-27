"""Batch processing of names using OpenAI API."""
import csv
import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

from common.config import DATASET_PATH, TASKS_DIR, BATCH_TRACKING_FILE, PROMPT
from common.schema import create_name_schema
from usage_tracker import tracker

# Initialize OpenAI client
load_dotenv()
client = OpenAI()


def generate_tasks(stop: int, start: int = 0) -> str:
    """
    Generates a JSONL file containing OpenAI API tasks for batch processing.

    Args:
        stop (int): Number of records to process
        start (int): Starting index (default: 0)

    Returns:
        str: Path to the generated tasks file
    """
    # Create tasks directory if it doesn't exist
    if not os.path.exists(TASKS_DIR):
        os.makedirs(TASKS_DIR)
        
    # Generate filename with range
    tasks_path = os.path.join(TASKS_DIR, f"tasks_{start:05d}-{stop:05d}.jsonl")
    
    with (
        open(DATASET_PATH, "r", encoding="utf-8-sig") as src,
        open(tasks_path, "w") as dst,
    ):
        reader = itertools.islice(csv.DictReader(src), start, stop)
        for row in reader:
            name = row["CUSTDES"]
            response_format = create_name_schema(name)
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
                                {"role": "user", "content": name},
                            ],
                            "response_format": response_format,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    
    return tasks_path


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
    Finds the batch ID for a given range from the tracking file.
    
    Args:
        start (int): Start of the range
        stop (int): End of the range
        
    Returns:
        str: Batch ID if found, None otherwise
    """
    range_str = f"{start:05d}-{stop:05d}"
    if not os.path.exists(BATCH_TRACKING_FILE):
        return None
        
    with open(BATCH_TRACKING_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                range_part, batch_id, *_ = line.strip().split("|")
                if range_part == range_str:
                    return batch_id
    return None


def create_batch(stop: int, start: int = 0):
    """
    Creates a new batch processing job with OpenAI's API.

    Args:
        stop (int): Number of records to process
        start (int): Starting index (default: 0)
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
    Formats batch job status in a readable way, including token usage and cost.
    
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
    
    for line in content.decode('utf-8').splitlines():
        if not line.strip():
            continue
            
        data = json.loads(line)
        
        try:
            # Get the original name safely from the request
            messages = data.get('request', {}).get('body', {}).get('messages', [])
            original_name = messages[1]['content'] if len(messages) > 1 else 'unknown'
            
            response_data = data['response']['body']['choices'][0]['message']['content']
            name_data = json.loads(response_data)
            
            if name_data['is_person']:
                successful_results.append({
                    'id': data['custom_id'],
                    'given_name': name_data['given_name'],
                    'family_name': name_data['family_name']
                })
            else:
                rejected_results.append({
                    'id': data['custom_id'],
                    'name': original_name,
                    'reason': 'not_a_person'
                })
        except Exception as e:
            # Try to get the original name from different possible locations in the data
            try:
                messages = data.get('request', {}).get('body', {}).get('messages', [])
                original_name = messages[1]['content'] if len(messages) > 1 else data.get('custom_name', 'unknown')
            except:
                original_name = data.get('custom_name', 'unknown')
                
            rejected_results.append({
                'id': data['custom_id'],
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
        reject_file = open(reject_path, "w", newline="", encoding="utf-8")
        
        result_writer = csv.DictWriter(result_file, fieldnames=["id", "given_name", "family_name"])
        reject_writer = csv.DictWriter(reject_file, fieldnames=["id", "name", "reason"])
        
        result_writer.writeheader()
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