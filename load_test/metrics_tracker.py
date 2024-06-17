import asyncio
import time
from typing import Dict, Any, Union, List
import numpy as np


class MetricsTracker:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.start_time = time.time()
        self.test_complete = False
        self.response_times: List[float] = []
        self.metrics: Dict[str, Union[int, float]] = {
            "active_calls": 0,
            "successful_calls": 0,
            "unsuccessful_calls": 0,
            "total_calls": 0,
            "max_concurrent_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_token_count": 0,
            "rate_limit_calls": 0,
            "avg_response_time": 0,
        }

    async def update_metric(self, metric_name: str, value: int):
        async with self.lock:
            if metric_name in self.metrics:
                if metric_name == "active_calls":
                    self.metrics["max_concurrent_calls"] = max(
                        self.metrics["max_concurrent_calls"],
                        self.metrics["active_calls"],
                    )
                    self.metrics["active_calls"] += value
                elif metric_name == "avg_response_time":
                    self.metrics["avg_response_time"] = (
                        self.metrics["avg_response_time"]
                        * self.metrics["successful_calls"]
                        + value
                    ) / (self.metrics["successful_calls"] + 1)
                    self.metrics["successful_calls"] += 1
                    self.response_times.append(value)
                elif metric_name == "total_token_count":
                    # self.metrics["total_output_tokens"] = (
                    #     (self.metrics["total_token_count"] + value)
                    #     - self.metrics["total_input_tokens"]
                    # )
                    self.metrics["total_token_count"] += value
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
            elapsed_time = time.time() - self.start_time
            elapsed_min = elapsed_time / 60

            # limit to 2 decimal places

            tokens_per_minute = int(
                (self.metrics["total_output_tokens"] / elapsed_min)
                if elapsed_time > 0
                else 0
            )
            requests_per_minute = int(
                (self.metrics["successful_calls"] / elapsed_min)
                if elapsed_time > 0
                else 0
            )

            if len(self.response_times) == 0:
                return dict(
                    self.metrics,
                    p50=0,
                    p90=0,
                    p99=0,
                    tokens_per_minute=tokens_per_minute,
                    requests_per_minute=requests_per_minute,
                )

            p50 = np.percentile(self.response_times, 50).round(3)
            p90 = np.percentile(self.response_times, 90).round(3)
            p99 = np.percentile(self.response_times, 99).round(3)

            return dict(
                self.metrics,
                p50=p50,
                p90=p90,
                p99=p99,
                tokens_per_minute=tokens_per_minute,
                requests_per_minute=requests_per_minute,
            )

    async def set_test_complete(self, value: bool = True) -> None:
        self.test_complete = value

    async def is_test_complete(self) -> bool:
        return self.test_complete
