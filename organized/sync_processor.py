"""Synchronous processing of names using OpenAI API."""
import asyncio
import csv
import itertools
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv

from common.config import DATASET_PATH, RESULT_DIR_PATH, PROMPT
from common.schema import create_name_schema
from common.validators import is_name_valid, validate_name_parts
from common.writers import setup_writers
from usage_tracker import tracker
from throttler import throttle

# Initialize OpenAI client
load_dotenv()
async_client = AsyncOpenAI()


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
        response_format=create_name_schema(name),
        temperature=0.15,
        max_completion_tokens=128,
    )

    tracker.add_usage(response, model="gpt-4o-mini")
    return response


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
        start, stop, RESULT_DIR_PATH)

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