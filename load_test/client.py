import httpx
from typing import Dict, Any, Optional, Literal
from openai import AsyncOpenAI, AsyncAzureOpenAI, APIStatusError, APIError
from .metrics_tracker import MetricsTracker
import tiktoken
import time
from logging import Logger

try:
    from .custom_handler import CustomClient
except ImportError:
    CustomClient = None


class AsyncClient:
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        metrics_tracker: MetricsTracker,
        logger: Logger,
        max_tokens: int = None,
        api_version: str = "2023-05-15",
        client_type: Literal["azure", "openai", "custom"] = "azure",
        tiktoken_encoding: str = "cl100k_base",
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.max_tokens = max_tokens
        self.metrics_tracker = metrics_tracker
        if tiktoken_encoding not in tiktoken.list_encoding_names():
            raise ValueError(f"Unsupported TikToken encoding: {tiktoken_encoding}")
        self.tiktoken = tiktoken.get_encoding(tiktoken_encoding)
        self.logger = logger
        self.client_type = client_type
        match client_type:
            case "azure":
                self.client = AsyncAzureOpenAI(
                    base_url=self.endpoint,
                    api_key=self.api_key,
                    api_version=self.api_version,
                )
            case "openai":
                self.client = AsyncOpenAI(
                    base_url=self.endpoint,
                    api_key=self.api_key,
                )
            case "custom":
                if CustomClient is None:
                    raise ValueError("CustomClient is not defined.")
                self.client = CustomClient(
                    endpoint=self.endpoint,
                    api_key=self.api_key,
                    tiktoken_encoding=self.tiktoken,
                )
            case default:
                raise ValueError(f"Unsupported client type: {client_type}")

    async def chat_completions(
        self,
        model: str,
        message: str,
    ) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": self.max_tokens,
        }
        token_count = len(self.tiktoken.encode(message))
        await self.metrics_tracker.update_metric("total_input_tokens", token_count)
        start_time = time.perf_counter()

        try:
            self.logger.info(f"Sending request with payload. id: {id(payload)}")
            await self.metrics_tracker.update_metric("active_calls", 1)

            if self.client_type == "custom":
                response = await self.client.custom_request_handler(
                    model, message, self.max_tokens
                )
            else:
                response = await self.client.chat.completions.create(
                    model=payload["model"],
                    messages=payload["messages"],
                    max_tokens=payload["max_tokens"],
                )
            self.logger.info(f"Received response from API. id: {id(payload)}")
            end_time = time.perf_counter()
            response_time = end_time - start_time
            await self.metrics_tracker.update_metric("avg_response_time", response_time)
            token_count = (
                response.usage.total_tokens
                if self.client_type != "custom"
                else self.client.custom_response_handler(response)
            )
            if type(token_count) != int:
                raise ValueError(f"Unsupported token count type: {type(token_count)}")
            await self.metrics_tracker.update_metric(
                "sucessful_token_count", token_count + token_count
            )

        except (httpx.HTTPStatusError, APIError) as e:

            if e.response.status_code == 429:
                # Rate limit exceeded
                await self.metrics_tracker.update_metric("rate_limit_calls", 1)
            self.logger.warning("Error here")
            await self.metrics_tracker.update_metric("unsuccessful_calls", 1)
            self.logger.error(
                f"Received error response from API. id: {id(payload)} status_code: {e.response.status_code} response: {e.response.json()}"
            )
            raise e
        finally:
            await self.metrics_tracker.update_metric("active_calls", -1)
            await self.metrics_tracker.update_metric("total_calls", 1)
