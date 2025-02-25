import asyncio
import csv
import itertools
from typing import TextIO, Tuple
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import argparse

from config import Config
from name_processor import NameProcessor, NameProcessingResult
from usage_tracker import tracker
from batch_processor import BatchProcessor

def setup_writers(start: int, stop: int) -> Tuple[csv.DictWriter, csv.DictWriter, TextIO, TextIO]:
    """Sets up CSV writers for results and rejections."""
    result_path = os.path.join(Config.RESULT_DIR_PATH, f"{start:05d}-{(start + stop):05d}.csv")
    reject_path = os.path.join(Config.RESULT_DIR_PATH, f"{start:05d}-{(start + stop):05d}_rejects.csv")
    
    result_file = open(result_path, "w", newline="", encoding="utf-8")
    reject_file = open(reject_path, "w", newline="", encoding="utf-8")
    
    result_writer = csv.DictWriter(result_file, fieldnames=["id", "first_name", "last_name"])
    reject_writer = csv.DictWriter(reject_file, fieldnames=["id", "name", "reason"])
    
    result_writer.writeheader()
    reject_writer.writeheader()
    
    return result_writer, reject_writer, result_file, reject_file

async def process_results(results: list[NameProcessingResult], 
                         result_writer: csv.DictWriter, 
                         reject_writer: csv.DictWriter) -> None:
    """Write processing results to appropriate files."""
    for result in results:
        if not result.is_valid:
            reject_writer.writerow({
                "id": result.id,
                "name": result.first_name,  # Original name stored in first_name when invalid
                "reason": result.error_message
            })
        else:
            result_writer.writerow({
                "id": result.id,
                "first_name": result.first_name,
                "last_name": result.last_name
            })

async def main(stop: int, start: int = 0) -> None:
    """
    Main processing function that handles the name processing pipeline.
    
    Args:
        stop (int): Number of records to process
        start (int): Starting index (default: 0)
    """
    # Initialize
    load_dotenv()
    client = AsyncOpenAI()
    processor = NameProcessor(Config, client)
    
    # Setup writers
    writer, reject_writer, result_file, reject_file = setup_writers(start, stop)
    
    try:
        with open(Config.DATASET_PATH, "r", encoding="utf-8-sig") as src:
            reader = itertools.islice(csv.DictReader(src), start, stop)
            
            while True:
                # Process in chunks of 50
                chunk = list(itertools.islice(reader, 50))
                if not chunk:
                    break
                    
                # Process chunk and write results
                results = await processor.process_chunk(chunk)
                await process_results(results, writer, reject_writer)
                
                # Ensure data is written to files
                result_file.flush()
                reject_file.flush()
        
        # Print token usage summary
        tracker.print_summary()
        
    finally:
        # Clean up
        result_file.close()
        reject_file.close()

if __name__ == "__main__":
    # Example usage with command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["stream", "batch"], default="stream")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=100)
    parser.add_argument("--batch-id", type=str)
    args = parser.parse_args()

    if args.mode == "stream":
        asyncio.run(main(args.stop, args.start))
    else:
        client = AsyncOpenAI()
        batch_processor = BatchProcessor(client)
        if args.batch_id:
            batch_processor.get_batch(args.batch_id)
        else:
            batch_processor.generate_tasks(args.stop - args.start)
            batch_processor.create_batch()