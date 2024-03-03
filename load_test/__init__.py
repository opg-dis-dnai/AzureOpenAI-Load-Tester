import asyncio
from .parse_args import parse, CommandLineArgs
from .util import setup_logging, parse_duration, generate_template_string, prompt_looper
from .prompts import prompts
from .client import AsyncClient
from .live_monitor import LiveMonitor
from .metrics_tracker import MetricsTracker


async def run_test(
    args: CommandLineArgs,
    client: AsyncClient,
    live_monitor: LiveMonitor,
    metrics_tracker: MetricsTracker,
    message: str,
):
    # Convert duration string to timedelta
    duration = parse_duration(args.duration)
    end_time = (
        asyncio.get_running_loop().time() + duration.total_seconds()
        if duration
        else None
    )

    async def perform_request():
        # Generate a test string or payload here as needed
        await client.chat_completions(model=args.model, message=message)

    async def manage_requests():
        tasks = set()
        try:
            while end_time is None or asyncio.get_running_loop().time() < end_time:
                if tasks:
                    done, _ = await asyncio.wait(
                        tasks, timeout=0, return_when=asyncio.FIRST_COMPLETED
                    )
                    tasks.difference_update(done)

                while len(tasks) < args.concurrency_level and (
                    end_time is None or asyncio.get_running_loop().time() < end_time
                ):
                    task = asyncio.create_task(perform_request())
                    tasks.add(task)

                await asyncio.sleep(1)  # Adjust as necessary to manage load
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await metrics_tracker.set_test_complete()

    # Start the test
    await asyncio.gather(manage_requests(), live_monitor.monitor_metrics())


async def main_async():
    # Parse command line arguments
    args = parse()

    # Setup logging
    logger = setup_logging()

    # Initialize the metrics tracker
    metrics_tracker = MetricsTracker()

    # Initialize the API client
    client = AsyncClient(
        endpoint=args.endpoint,
        api_key=args.api_key,
        metrics_tracker=metrics_tracker,
        logger=logger,
        max_tokens=args.max_tokens,
        tiktoken_encoding=args.tiktoken,
        client_type=args.client_type,
        api_version=args.api_version,
    )

    # Initialize and start the live monitoring
    live_monitor = LiveMonitor(metrics_tracker)

    test_string = next(prompt_looper(prompts))

    # Run the test
    await run_test(args, client, live_monitor, metrics_tracker, message=test_string)

    # Final update to the live monitor
    await live_monitor.final_update()
