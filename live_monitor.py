import asyncio
from rich.console import Console
from rich.table import Table
from rich.live import Live
from typing import Optional

from metrics_tracker import MetricsTracker


class LiveMonitor:
    def __init__(
        self, metrics_tracker: MetricsTracker, console: Optional[Console] = None
    ) -> None:
        self.metrics_tracker = metrics_tracker
        self.console = console or Console()
        self.live = Live(console=self.console, refresh_per_second=4)

    def create_table(self) -> Table:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim", width=20)
        table.add_column("Value")
        return table

    async def update_table(self, table: Table) -> None:
        metrics = await self.metrics_tracker.get_metrics()
        table.rows = []  # Clear existing rows
        for key, value in metrics.items():
            table.add_row(key.replace("_", " ").title(), str(value))

    async def monitor_metrics(self, update_interval: int = 1) -> None:
        with self.live as live:
            while not await self.metrics_tracker.is_test_complete():
                table = self.create_table()
                await self.update_table(table)
                live.update(table)
                await asyncio.sleep(update_interval)

    async def final_update(self) -> None:
        # This is called once the test is complete to do a final update of the table
        table = self.create_table()
        await self.update_table(table)
        self.live.update(table)
