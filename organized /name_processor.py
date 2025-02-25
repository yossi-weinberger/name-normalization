from dataclasses import dataclass
from typing import Optional, List, Dict
import asyncio
import json
from openai import AsyncOpenAI
from throttler import throttle
from validators import is_name_valid, validate_name_parts
from usage_tracker import tracker

@dataclass
class NameProcessingResult:
    id: str
    first_name: Optional[str]
    last_name: Optional[str]
    is_valid: bool
    error_message: Optional[str]

class NameProcessor:
    def __init__(self, config, client: AsyncOpenAI):
        self.config = config
        self.client = client

    @throttle(rate_limit=500, period=60)
    async def process_single_name(self, name: str):
        """Process a single name using OpenAI's API"""
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.config.SYSTEM_PROMPT},
                {"role": "user", "content": name}
            ],
            response_format=self.config.RESPONSE_FORMAT,
            temperature=0.15,
            max_completion_tokens=128,
        )
        
        tracker.add_usage(response, model="gpt-4o-mini")
        return response

    async def process_chunk(self, chunk: List[Dict]) -> List[NameProcessingResult]:
        """Process a chunk of names and return results"""
        tasks = []
        results = []
        
        # First, validate names and create tasks
        for row in chunk:
            if not is_name_valid(row["CUSTDES"]):
                results.append(NameProcessingResult(
                    id=row["CUST"],
                    first_name=row["CUSTDES"],  # Store original name
                    last_name=None,
                    is_valid=False,
                    error_message="invalid_characters"
                ))
            else:
                tasks.append((
                    row["CUST"],
                    row["CUSTDES"],
                    self.process_single_name(row["CUSTDES"])
                ))
        
        # Process valid names with API
        responses = await asyncio.gather(*[task[2] for task in tasks])
        
        # Handle API responses
        for (id, full_name), response in zip([(t[0], t[1]) for t in tasks], responses):
            try:
                parts = json.loads(response.choices[0].message.content)
                
                if not parts["is_person"]:
                    results.append(NameProcessingResult(
                        id=id,
                        first_name=full_name,  # Store original name
                        last_name=None,
                        is_valid=False,
                        error_message="not_a_person"
                    ))
                    continue
                
                # Validate name parts
                is_valid, error_message = validate_name_parts(
                    full_name,
                    parts["first_name"],
                    parts["last_name"]
                )
                
                if not is_valid:
                    results.append(NameProcessingResult(
                        id=id,
                        first_name=full_name,  # Store original name
                        last_name=None,
                        is_valid=False,
                        error_message=error_message
                    ))
                    continue
                
                # Add valid result
                results.append(NameProcessingResult(
                    id=id,
                    first_name=parts["first_name"],
                    last_name=parts["last_name"],
                    is_valid=True,
                    error_message=None
                ))
                
            except Exception as e:
                results.append(NameProcessingResult(
                    id=id,
                    first_name=full_name,  # Store original name
                    last_name=None,
                    is_valid=False,
                    error_message=f"processing_error: {str(e)}"
                ))
        
        return results 