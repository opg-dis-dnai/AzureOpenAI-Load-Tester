import logging
from datetime import datetime, timedelta
import re
import argparse
from typing import Optional
import wonderwords
import tiktoken


# Logging configuration utility
def setup_logging(log_file_prefix: str = "test", level: int = logging.INFO) -> None:
    filename = f"{log_file_prefix}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=filename,
    )


# Utility to parse the duration string into a timedelta object
def parse_duration(duration_str: Optional[str]) -> Optional[timedelta]:
    if duration_str is None:
        return None  # Indefinite duration

    match = re.match(r"(\d+)(s|m|h)", duration_str)
    if not match:
        raise argparse.ArgumentTypeError(
            "Invalid duration format. Use 's' for seconds, 'm' for minutes, and 'h' for hours. E.g., '60s', '2m', '1h'."
        )

    duration, unit = match.groups()
    duration = int(duration)

    if unit == "s":
        return timedelta(seconds=duration)
    elif unit == "m":
        return timedelta(minutes=duration)
    elif unit == "h":
        return timedelta(hours=duration)
    else:
        raise ValueError(f"Unsupported time unit: {unit}")


# Function to generate a test string with a target token count
def generate_test_string(target_token_count: int):
    r = wonderwords.RandomWord()
    encoding = tiktoken.get_encoding("cl100k_base")

    string_builder = ""

    avg_tokens_per_word = 1.3  # Googled it
    estimated_word_count = int(target_token_count / avg_tokens_per_word)

    # Generate the estimated number of words
    words = [
        r.word(include_parts_of_speech=["adjectives", "nouns"])
        for _ in range(estimated_word_count)
    ]

    string_builder = " ".join(words)

    # Refine the string to meet the exact token count
    while True:
        token_count = len(encoding.encode(string_builder))

        if token_count < target_token_count:
            string_builder += " " + r.word(
                include_parts_of_speech=["adjectives", "nouns"]
            )

        elif token_count > target_token_count:
            string_builder = " ".join(string_builder.split(" ")[:-1])

        else:
            break

    return string_builder
