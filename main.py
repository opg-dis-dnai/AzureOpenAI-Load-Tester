import asyncio
from parse_args import parse_args, CommandLineArgs
from util import setup_logging, parse_duration, generate_test_string
from client import AsyncAzureOpenAIClient
from live_monitor import LiveMonitor
from metrics_tracker import MetricsTracker


async def run_test(
    args: CommandLineArgs,
    client: AsyncAzureOpenAIClient,
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
        while end_time is None or asyncio.get_running_loop().time() < end_time:
            tasks = [
                asyncio.create_task(perform_request())
                for _ in range(args.concurrency_level)
            ]
            await asyncio.gather(*tasks)
            # Optional: add a slight delay if needed to manage rate limits or server load
            await asyncio.sleep(1)  # Adjust as necessary

    # Start the test
    await asyncio.gather(manage_requests(), live_monitor.monitor_metrics())

    # Mark test as complete to stop the live monitoring
    await metrics_tracker.set_test_complete()


async def main_async():
    # Parse command line arguments
    args = parse_args()

    # Setup logging
    setup_logging()

    # Initialize the metrics tracker
    metrics_tracker = MetricsTracker()

    # Initialize the API client
    client = AsyncAzureOpenAIClient(
        azure_endpoint=args.endpoint,
        api_key=args.api_key,
        metrics_tracker=metrics_tracker,
        max_tokens=args.max_tokens,
    )

    # Initialize and start the live monitoring
    live_monitor = LiveMonitor(metrics_tracker)

    test_string = generate_test_string(args.token_count)

    # Run the test
    await run_test(args, client, live_monitor, metrics_tracker, message=test_string)

    # Final update to the live monitor
    live_monitor.final_update()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
