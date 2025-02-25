import json
import csv
import itertools
from openai import OpenAI
from config import Config

class BatchProcessor:
    def __init__(self, client: OpenAI):
        self.client = client

    def generate_tasks(self, top: int) -> None:
        """
        Generates a JSONL file containing OpenAI API tasks for batch processing.
        
        Args:
            top (int): Number of names to process from the dataset
        """
        with (
            open(Config.DATASET_PATH, "r", encoding="utf-8-sig") as src,
            open(Config.TASKS_PATH, "w") as dst,
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
                                        "content": Config.SYSTEM_PROMPT,
                                    },
                                    {"role": "user", "content": row["CUSTDES"]},
                                ],
                                "response_format": Config.RESPONSE_FORMAT,
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def create_batch(self):
        """Creates a new batch processing job with OpenAI's API."""
        batch_file = self.client.files.create(
            file=open(Config.TASKS_PATH, "rb"),
            purpose="batch",
        )

        batch_job = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        print(batch_job)

    def get_batch(self, batch_id: str):
        """
        Retrieves and processes the results of a batch job.
        
        Args:
            batch_id (str): The ID of the batch job to retrieve
        """
        batch_job = self.client.batches.retrieve(batch_id)
        print(batch_job)

        if batch_job.status == "completed":
            print("Batch is completed")
            content = self.client.files.content(batch_job.output_file_id).content
            with open("results.jsonl", "wb") as f:
                f.write(content) 