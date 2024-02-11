import httpx
from typing import Dict, Any, Optional
from openai import AsyncAzureOpenAI
from metrics_tracker import MetricsTracker
import tiktoken
import time


class AsyncAzureOpenAIClient:
    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        metrics_tracker: MetricsTracker,
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
        await self.metrics_tracker.update_metric("active_calls", 1)
        token_count = len(self.tiktoken.encode(message))
        start_time = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(
                model=payload["model"],
                messages=payload["messages"],
                max_tokens=payload["max_tokens"],
            )
            end_time = time.perf_counter()
            response_time = end_time - start_time
            await self.metrics_tracker.update_metric("avg_response_time", response_time)
            await self.metrics_tracker.update_metric("total_tokens", token_count)

        except httpx.HTTPStatusError as e:

            if e.response.status_code == 429:
                # Rate limit exceeded
                await self.metrics_tracker.update_metric("rate_limit_calls", 1)

            await self.metrics_tracker.update_metric("unsuccessful_calls", 1)
        finally:
            await self.metrics_tracker.update_metric("active_calls", -1)
