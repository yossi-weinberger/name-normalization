"""File writing utilities for name processing."""
import csv
import os
import time
from typing import Tuple, Dict


def setup_writers(start: int, stop: int, result_dir_path: str, append_rejects: bool = False) -> Tuple[csv.DictWriter, csv.DictWriter, object, object]:
    """
    Sets up CSV writers for results and rejections.

    Args:
        start (int): Starting index for file naming
        stop (int): Ending index for file naming
        result_dir_path (str): Path to the directory where results should be saved
        append_rejects (bool): Whether to append to the rejects file (default: False)

    Returns:
        tuple: (result_writer, reject_writer, result_file, reject_file)
    """
    range_str = f"{start:05d}-{stop:05d}"
    result_path = os.path.join(result_dir_path, f"results_{range_str}.csv")
    reject_path = os.path.join(result_dir_path, f"results_{range_str}_rejects.csv")

    result_file = open(result_path, "w", newline="", encoding="utf-8")
    reject_file = open(reject_path, "a" if append_rejects else "w", newline="", encoding="utf-8")

    result_writer = csv.DictWriter(result_file, fieldnames=["id", "given_name", "family_name"])
    reject_writer = csv.DictWriter(reject_file, fieldnames=["id", "name", "reason"])

    result_writer.writeheader()
    if not append_rejects or reject_file.tell() == 0:
        reject_writer.writeheader()

    return result_writer, reject_writer, result_file, reject_file


def write_status_log(batch_id: str, batch_status: str, request_counts: Dict, usage_data: Dict, range_str: str, result_dir_path: str) -> None:
    """
    Writes batch processing status to a log file.

    Args:
        batch_id (str): The ID of the batch job
        batch_status (str): Current status of the batch job
        request_counts (Dict): Dictionary containing request count information
        usage_data (Dict): Dictionary containing API usage information
        range_str (str): String representing the range of records being processed
        result_dir_path (str): Path to the directory where results should be saved
    """
    log_path = os.path.join(result_dir_path, f"batch_{range_str}_status_log.txt")
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
        f.write("=" * 50 + "\n")
        f.write(f"Task ID: {batch_id}\n")
        f.write(f"Status: {batch_status}\n")
        f.write(f"Progress: {request_counts['completed']}/{request_counts['total']} completed\n")
        f.write(f"Failed: {request_counts['failed']}\n\n")
        
        f.write("Usage Information:\n")
        f.write(f"API Calls: {usage_data['total_calls']}\n")
        f.write(f"Input Tokens: {usage_data['input_tokens']:,}\n")
        f.write(f"Output Tokens: {usage_data['output_tokens']:,}\n")
        f.write(f"Total Tokens: {usage_data['total_tokens']:,}\n")
        f.write(f"Total Cost: ${usage_data['total_cost']:.4f}\n")
        f.write("=" * 50 + "\n\n") 