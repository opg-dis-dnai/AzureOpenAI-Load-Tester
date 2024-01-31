import argparse
import asyncio
import time
import logging
from rich.console import Console
from rich.table import Table
from rich.live import Live
import re
from datetime import timedelta
import os
from openai import AsyncAzureOpenAI
from typing import Optional
import wonderwords
import tiktoken
import httpx


def duration_type(duration_str):
    match = re.match(r"(\d+)(s|m|h)", duration_str)

    if not match:
        raise argparse.ArgumentTypeError("Invalid duration format")

    return duration_str


# Argument parsing
parser = argparse.ArgumentParser(description="Azure OpenAI Test Harness")
parser.add_argument(
    "-e", "--endpoint", type=str, required=True, help="Azure OpenAI API endpoint"
)
parser.add_argument(
    "-k", "--api-key", type=str, required=True, help="Azure OpenAI API key"
)
parser.add_argument(
    "-m", "--model", type=str, required=True, help="Azure OpenAI deployment name"
)
parser.add_argument(
    "-c",
    "--concurrency-level",
    type=int,
    default=1,
    help="Level of concurrency for testing",
)
parser.add_argument(
    "-t", "--token-count", type=int, default=100, help="Desired token count per request"
)
parser.add_argument(
    "-d",
    "--duration",
    type=duration_type,
    default=None,
    help="Duration of the test (e.g., '60s', '2m', '1h'). If not set, runs indefinitely.",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Function to parse the duration string
def parse_duration(duration_str):
    if duration_str is None:
        return None  # Indefinite duration

    match = re.match(r"(\d+)(s|m|h)", duration_str)
    if not match:
        raise ValueError(
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


# Metrics tracker class
class MetricsTracker:
    def __init__(self):
        self.active_calls = 0
        self.active_successful_calls = 0  # Track successful calls separately
        self.max_concurrent_calls = 0
        self.successful_calls = 0
        self.unsuccessful_calls = 0
        self.max_concurrent_successful_calls = (
            0  # Track max concurrent successful calls
        )
        self.total_tpm = 0
        self.start_time = time.time()
        self.rate_limit = 0
        self.total_token = 0

    def update_max_concurrent(self):
        self.max_concurrent_calls = max(self.max_concurrent_calls, self.active_calls)

    def update_max_concurrent_successful(self):
        self.max_concurrent_successful_calls = max(
            self.max_concurrent_successful_calls, self.active_successful_calls
        )

    def update_tpm(self, token_count: int):
        self.total_token += token_count  # Accumulate the tokens
        print(self.total_token)
        elapsed_time = time.time() - self.start_time
        self.total_tpm = (self.total_tokens / elapsed_time) * 60


# Async function to simulate an API call
async def test_api_call(
    model: str,
    api_key: str,
    token_count: int,
    metrics: MetricsTracker,
    client: AsyncAzureOpenAI,
    test_string: str,
):
    # Update active calls
    metrics.active_calls += 1
    metrics.update_max_concurrent()

    try:
        # Placeholder for actual API call logic
        response = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": test_string}]
        )
        metrics.active_successful_calls += 1
        metrics.successful_calls += 1
        metrics.update_tpm(token_count)
        metrics.update_max_concurrent_successful()  # Update max concurrent successful calls
    except httpx.HTTPStatusError as e:
        metrics.unsuccessful_calls += 1

        if e.response.status_code == 429:
            metrics.rate_limit += 1

    except Exception as e:
        logging.error(f"\nAPI call failed: {e}\n")

        metrics.unsuccessful_calls += 1
    finally:
        metrics.active_calls -= 1


# Main async function to perform tests
async def perform_tests(
    api_key: str,
    concurrency_level: int,
    token_count: int,
    model: str,
    duration: Optional[timedelta],
    metrics: MetricsTracker,
    client: AsyncAzureOpenAI,
    test_string: str,
):
    semaphore = asyncio.Semaphore(concurrency_level)
    start_time = time.time()
    tasks = []

    async def semaphored_test_api_call(*args, **kwargs):
        async with semaphore:
            await test_api_call(*args, **kwargs)

    while True:
        if duration and time.time() - start_time > duration.total_seconds():
            break

        if len(tasks) < concurrency_level:
            new_task = asyncio.create_task(
                semaphored_test_api_call(
                    model, api_key, token_count, metrics, client, test_string
                )
            )
            tasks.append(new_task)
        else:
            # Wait for any task to be completed
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            tasks = [task for task in tasks if not task.done()]

    # Await any remaining tasks
    if tasks:
        await asyncio.gather(*tasks)


def update_live_table(metrics: MetricsTracker, table: Table):
    # Update the table with the latest metrics

    table.rows[0].cells[1].text = str(metrics.successful_calls)

    table.rows[1].cells[1].text = str(metrics.unsuccessful_calls)

    table.rows[2].cells[1].text = str(metrics.active_calls)

    table.rows[3].cells[1].text = str(metrics.max_concurrent_calls)

    table.rows[4].cells[1].text = f"{metrics.total_tpm:.2f}"

    table.rows[5].cells[1].text = str(metrics.rate_limit)

    table.rows[6].cells[1].text = str(metrics.total_token)

    table.rows[7].cells[1].text = str(metrics.active_successful_calls)

    table.rows[8].cells[1].text = str(metrics.max_concurrent_successful_calls)


async def monitor_metrics(metrics: MetricsTracker, table: Table, interval: int):
    while True:
        update_live_table(metrics, table)

        await asyncio.sleep(interval)


async def async_main(args, console, table, metrics, test_string, client):
    duration: Optional[timedelta] = parse_duration(args.duration)

    # Set the update interval for the live table in seconds

    update_interval = 1

    monitor_task = asyncio.create_task(monitor_metrics(metrics, table, update_interval))

    try:
        await perform_tests(
            args.api_key,
            args.concurrency_level,
            args.token_count,
            args.model,
            duration,
            metrics,
            client,
            test_string,
        )

    finally:
        monitor_task.cancel()

    # Final update to the table before exiting Live context

    update_live_table(metrics, table)


# Main entry point
def main():
    args = parser.parse_args()
    duration: Optional[timedelta] = parse_duration(args.duration)  # Parse duration
    client = AsyncAzureOpenAI(
        azure_endpoint=args.endpoint, api_key=args.api_key, api_version="2023-05-15"
    )
    metrics = MetricsTracker()
    test_string = generate_test_string(args.token_count)

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=20)
    table.add_column("Value")

    # Add rows for each metric in metric tracker
    table.add_row("Successful Calls", "0")
    table.add_row("Unsuccessful Calls", "0")
    table.add_row("Active Calls", "0")
    table.add_row("Max Concurrent Calls", "0")
    table.add_row("Total TPM", "0")
    table.add_row("Rate Limit", "0")
    table.add_row("Total Token", "0")
    table.add_row("Active Successful Calls", "0")
    table.add_row("Max Concurrent Successful Calls", "0")

    update_interval = 1

    with Live(table, console=console, refresh_per_second=10) as live:
        asyncio.run(async_main(args, console, table, metrics, test_string, client))
        # monitor_task = asyncio.create_task(
        #     monitor_metrics(metrics, table, update_interval)
        # )

        # asyncio.run(
        #     perform_tests(
        #         args.api_key,
        #         args.concurrency_level,
        #         args.token_count,
        #         args.model,
        #         duration,
        #         metrics,
        #         client,
        #         test_string,
        #     )
        # )

        # # Once the tests are done, cancel the monitoring task
        # monitor_task.cancel()

        # # Final update to the table before exiting Live context

        # update_live_table(metrics, table)

    console.print(table)


if __name__ == "__main__":
    main()
