import asyncio
import time
from typing import Dict, Any


class MetricsTracker:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.metrics: Dict[str, Any] = {
            "active_calls": 0,
            "successful_calls": 0,
            "unsuccessful_calls": 0,
            "max_concurrent_calls": 0,
            "total_tokens": 0,
            "start_time": time.time(),
            "test_complete": False,
            "tpm": 0,
            "rate_limit_calls": 0,
        }

    async def update_metric(self, metric_name: str, value: int):
        async with self.lock:
            if metric_name in self.metrics:

                if metric_name == "active_calls":
                    self.metrics["max_concurrent_calls"] = max(
                        self.metrics["max_concurrent_calls"],
                        self.metrics["active_calls"],
                    )
                elif metric_name == "total_tokens":
                    self.metrics["total_tokens"] += value
                else:
                    self.metrics[metric_name] += value

            else:
                raise KeyError(f"Metric {metric_name} does not exist.")

    async def set_metric(self, metric_name: str, value: Any):
        async with self.lock:
            if metric_name in self.metrics:
                self.metrics[metric_name] = value
            else:
                raise KeyError(f"Metric {metric_name} does not exist.")

    async def get_metric(self, metric_name: str):
        async with self.lock:
            if metric_name in self.metrics:
                return self.metrics[metric_name]
            else:
                raise KeyError(f"Metric {metric_name} does not exist.")

    async def get_metrics(self) -> Dict[str, Any]:
        async with self.lock:
            elapsed_time = time.time() - self.metrics["start_time"]
            tokens_per_minute = (
                (self.metrics["total_tokens"] / elapsed_time) * 60
                if elapsed_time > 0
                else 0
            )
            return dict(self.metrics, tokens_per_minute=tokens_per_minute)

    async def set_test_complete(self, value: bool = True) -> None:
        await self.set_metric("test_complete", value)

    async def is_test_complete(self) -> bool:
        return await self.get_metric("test_complete")
