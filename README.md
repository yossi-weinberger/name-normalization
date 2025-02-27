# Name Normalization Project

## Overview
This project is designed to normalize Hebrew names using OpenAI's GPT models. It processes a large dataset of names, splitting them into given names and family names while handling various edge cases and validations specific to Hebrew names.

## Processing Approaches

### 1. Direct Processing Approach
This approach processes names in real-time, one by one or in small chunks:

#### Features:
- Real-time processing and immediate results
- Direct API calls to OpenAI
- Synchronous or asynchronous processing options
- Better for smaller datasets or when immediate results are needed
- More control over individual name processing

#### Implementation:
```python
async def process_name(name: str):
    response = await async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": name}
        ],
        response_format=RESPONSE_FORMAT,
        temperature=0.15
    )
    return response

async def process(stop: int, start: int = 0):
    writer, reject_writer, result_file, reject_file = setup_writers(start, stop)
    # Process names in chunks of 50
    with open(DATASET_PATH, "r", encoding="utf-8-sig") as src:
        reader = itertools.islice(csv.DictReader(src), start, stop)
        while True:
            chunk = list(itertools.islice(reader, 50))
            if not chunk:
                break
            await process_chunk(chunk, writer, reject_writer)
```

### 2. Batch Processing Approach
This approach uses OpenAI's batch processing API for handling large datasets:

#### Features:
- Efficient processing of large datasets
- Better cost optimization
- Automatic retry mechanisms
- Built-in rate limiting
- Asynchronous processing by default
- Better for processing thousands of names

#### Implementation:
```python
def generate_tasks(stop: int, start: int = 0):
    # Generate JSONL file with tasks
    tasks_path = os.path.join(TASKS_DIR, f"tasks_{start:05d}-{stop:05d}.jsonl")
    # Create tasks for batch processing
    
def create_batch(stop: int, start: int = 0):
    # Submit batch job to OpenAI
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    
def get_batch(stop: int, start: int = 0):
    # Retrieve and process batch results
    batch_job = client.batches.retrieve(batch_id)
    # Process and save results
```

## Features
- Processes Hebrew names from CSV input files
- Splits full names into given names and family names
- Validates Hebrew character usage
- Handles batch processing for large datasets
- Tracks API usage and costs
- Generates detailed reports and logs

## Technical Architecture

### Core Components

1. **Name Processing**
   - Uses OpenAI's GPT-4 API for name analysis
   - Implements strict JSON schema validation
   - Handles Hebrew-specific name patterns
   - Removes titles (ד״ר, הרב, פרופ׳, עו״ד, etc.)

2. **Batch Processing System**
   - Splits large datasets into manageable chunks
   - Supports asynchronous batch operations
   - Implements rate limiting and throttling
   - Tracks batch progress and status

3. **Validation System**
   - Validates Hebrew character usage
   - Ensures output names contain only words from input
   - Validates person vs non-person names
   - Implements strict schema validation

### Data Flow

1. **Input Processing**
   - Reads from CSV file (`lp_members.csv`)
   - Validates input names for Hebrew characters
   - Generates batch tasks in JSONL format

2. **Batch Operations**
   - Creates OpenAI batch processing jobs
   - Tracks batch status and progress
   - Manages API rate limits and quotas

3. **Results Processing**
   - Separates successful and rejected results
   - Generates detailed CSV reports
   - Tracks usage statistics and costs

## Usage

### Direct Processing

```python
# Process a small batch of names
asyncio.run(process(77, 0))
```

### Batch Processing

```python
# Example of processing 70,000 records in 7 chunks
for i in range(7):
    start = i * 10000
    stop = (i + 1) * 10000
    generate_tasks(stop, start)
    create_batch(stop, start)
    get_batch(stop, start)
```

## Output Files

1. **Results Files**
   - `results_XXXXX-XXXXX.csv`: Successful name normalizations
   - `results_XXXXX-XXXXX_rejects.csv`: Rejected entries with reasons
   - `batch_XXXXX-XXXXX_status_log.txt`: Detailed processing logs

2. **Status Tracking**
   - `batch_tracking.txt`: Tracks all batch operations
   - Usage statistics and cost reports

## Configuration

### Environment Setup
- Requires OpenAI API credentials
- Uses `.env` file for configuration
- Supports both sync and async OpenAI clients

### Response Format
```json
{
    "is_person": boolean,
    "given_name": string,
    "family_name": string
}
```

## Error Handling

The system handles various error cases:
- Invalid Hebrew characters
- Non-person names
- API errors
- Validation failures
- Processing errors

## Performance Optimization

- Implements rate limiting (500 requests/60 seconds)
- Uses batch processing for efficiency
- Supports async operations
- Implements smart chunking for large datasets

## Monitoring and Logging

- Detailed batch status tracking
- Usage statistics and cost tracking
- Processing progress monitoring
- Error logging and reporting

## Dependencies
- OpenAI API
- python-dotenv
- asyncio
- csv
- json
- os
- re

## Security Considerations
- API key management through environment variables
- Secure file handling
- Input validation
- Error handling and logging

## When to Use Each Approach

### Use Direct Processing When:
- Processing small datasets (less than 1000 names)
- Need immediate results
- Want to handle each name individually
- Need more control over the processing flow
- Testing or debugging specific cases

### Use Batch Processing When:
- Processing large datasets (thousands of names)
- Cost optimization is important
- Can wait for results (up to 24 hours)
- Need better error handling and retries
- Want to minimize API rate limit issues 