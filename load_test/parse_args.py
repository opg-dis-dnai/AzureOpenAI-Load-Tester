import argparse
from typing import NamedTuple, Optional


class CommandLineArgs(NamedTuple):
    endpoint: str
    api_key: str
    model: str
    tiktoken: str
    concurrency_level: int
    duration: Optional[str]
    max_tokens: Optional[int]
    client_type: str
    api_version: str


def parse() -> CommandLineArgs:
    parser = argparse.ArgumentParser(description="Azure OpenAI Test Harness")
    parser.add_argument(
        "-e", "--endpoint", type=str, required=True, help="LLM API endpoint"
    )
    parser.add_argument(
        "-k", "--api-key", type=str, required=False, help="LLM API key", default=""
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="LLM name or Azure OpenAI deployment name",
    )
    parser.add_argument(
        "-tik",
        "--tiktoken",
        type=str,
        required=False,
        help="TikToken encoding name",
        default="cl100k_base",
    )
    parser.add_argument(
        "-c",
        "--concurrency-level",
        type=int,
        default=1,
        help="Level of concurrency for testing",
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
    parser.add_argument(
        "-az",
        "--azure-openai",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Azure OpenAI API. Default client is Azure OpenAI.",
    )
    parser.add_argument(
        "-v",
        "--api-version",
        type=str,
        default="2023-05-15",
        help="API version for the Azure OpenAI API. Default is '2023-05-15'.",
    )
    parser.add_argument(
        "-oai",
        "--openai",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use an OpenAI compatible API. Default client is Azure OpenAI.",
    )
    parser.add_argument(
        "-cstm",
        "--custom",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a custom API, if this is selected then you have to provide the request structure in the input.json file and the response structure in the output.json file.",
    )

    args = parser.parse_args()

    if (
        args.azure_openai
        and args.openai
        or args.azure_openai
        and args.custom
        or args.openai
        and args.custom
    ):
        raise ValueError(
            "Only one of --azure-openai, --openai, or --custom can be set at a time"
        )

    client_type = "azure"
    if args.openai:
        client_type = "openai"
    elif args.custom:
        client_type = "custom"

    return CommandLineArgs(
        endpoint=args.endpoint,
        api_key=args.api_key,
        model=args.model,
        tiktoken=args.tiktoken,
        concurrency_level=args.concurrency_level,
        duration=args.duration,
        max_tokens=args.max_tokens,
        client_type=client_type,
        api_version=args.api_version,
    )
