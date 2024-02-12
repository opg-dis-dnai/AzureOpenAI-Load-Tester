import httpx
from typing import Dict, Any, Optional
from openai import AsyncAzureOpenAI
from metrics_tracker import MetricsTracker
import tiktoken
import time
from logging import Logger


class AsyncAzureOpenAIClient:
    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        metrics_tracker: MetricsTracker,
        logger: Logger,
        max_tokens: int = None,
        api_version: str = "2023-05-15",
    ) -> None:
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.max_tokens = max_tokens
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.client = AsyncAzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )
        self.metrics_tracker = metrics_tracker
        self.tiktoken = tiktoken.get_encoding("cl100k_base")
        self.logger = logger

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
            self.logger.info(
                f"Sending request to Azure OpenAI with payload. id: {id(payload)}"
            )
            await self.metrics_tracker.update_metric("active_calls", 1)
            response = await self.client.chat.completions.create(
                model=payload["model"],
                messages=payload["messages"],
                max_tokens=payload["max_tokens"],
            )
            self.logger.info(f"Received response from Azure OpenAI. id: {id(payload)}")
            end_time = time.perf_counter()
            response_time = end_time - start_time
            await self.metrics_tracker.update_metric("avg_response_time", response_time)
            await self.metrics_tracker.update_metric(
                "sucessful_token_count", token_count + response.usage.total_tokens
            )

        except httpx.HTTPStatusError as e:

            if e.response.status_code == 429:
                # Rate limit exceeded
                await self.metrics_tracker.update_metric("rate_limit_calls", 1)

            await self.metrics_tracker.update_metric("unsuccessful_calls", 1)
            self.logger.error(
                f"Received error response from Azure OpenAI. id: {id(payload)} status_code: {e.response.status_code} response: {e.response.json()}"
            )
            raise e
        finally:
            await self.metrics_tracker.update_metric("active_calls", -1)
            await self.metrics_tracker.update_metric("total_calls", 1)
