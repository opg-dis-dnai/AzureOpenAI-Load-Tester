import argparse
from typing import NamedTuple, Optional


class CommandLineArgs(NamedTuple):
    endpoint: str
    api_key: str
    model: str
    concurrency_level: int
    token_count: int
    duration: Optional[str]
    max_tokens: Optional[int]


def parse_args() -> CommandLineArgs:
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
        "-t",
        "--token-count",
        type=int,
        default=100,
        help="Desired token count per request",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=str,
        default=None,
        help="Duration of the test (e.g., '60s', '2m', '1h'). If not set, runs indefinitely.",
    )
    parser.add_argument(
        "-mx",
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens parameter for the Azure OpenAI API. If not set, None is sent",
    )

    args = parser.parse_args()

    return CommandLineArgs(
        endpoint=args.endpoint,
        api_key=args.api_key,
        model=args.model,
        concurrency_level=args.concurrency_level,
        token_count=args.token_count,
        duration=args.duration,
        max_tokens=args.max_tokens,
    )
