import asyncio
from parse_args import parse_args, CommandLineArgs
from util import setup_logging, parse_duration, generate_template_string, promptLooper
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
    args = parse_args()

    # Setup logging
    logger = setup_logging()

    # Initialize the metrics tracker
    metrics_tracker = MetricsTracker()

    # Initialize the API client
    client = AsyncAzureOpenAIClient(
        azure_endpoint=args.endpoint,
        api_key=args.api_key,
        metrics_tracker=metrics_tracker,
        logger=logger,
        max_tokens=args.max_tokens,
    )

    # Initialize and start the live monitoring
    live_monitor = LiveMonitor(metrics_tracker)

    # test_string = generate_template_string(args.token_count)
    # test_string = "Tell me a very long story"

    prompts = ["Explain the principles of quantum computing to someone with no background in physics. Start with the basics of quantum mechanics, how quantum computers differ from classical computers, and then delve into qubits, superposition, and entanglement. Provide examples of potential applications of quantum computing in various fields, such as cryptography, drug discovery, and climate modeling, highlighting the advantages over traditional computing methods.",
                "Trace the development of the Internet from its inception to the present day. Describe the technological advancements that made the Internet possible, including the transition from ARPANET to the modern global network. Discuss the impact of the Internet on communication, education, business, and politics, including both positive and negative aspects. Conclude with thoughts on the future of the Internet, considering emerging technologies like 5G, blockchain, and the Internet of Things (IoT).",
                "Provide an in-depth guide to the most commonly used machine learning algorithms. For each algorithm, explain its working principle, types of problems it solves best, and its advantages and limitations. Include supervised learning algorithms like linear regression and decision trees, unsupervised learning algorithms like k-means clustering, and neural networks, with a focus on deep learning. Use practical examples to illustrate how each algorithm is applied in real-world scenarios, from predictive analytics to image recognition.",
    ]

    test_string = next(promptLooper(prompts))

    # Run the test
    await run_test(args, client, live_monitor, metrics_tracker, message=test_string)

    # Final update to the live monitor
    await live_monitor.final_update()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
