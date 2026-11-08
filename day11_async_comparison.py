
import asyncio
import time


URLS: list[str] = [
    "https://example.com/page/1",
    "https://example.com/page/2",
    "https://example.com/page/3",
    "https://example.com/page/4",
    "https://example.com/page/5",
]


# VERSION 1: SEQUENTIAL
def fetch_url_sequential(url: str) -> str:
    print(f"Starting: {url}")
    time.sleep(1)
    print(f"Finished: {url}")

    return f"Fetched: {url}"


def run_sequentially() -> list[str]:
    results: list[str] = []

    for url in URLS:
        result = fetch_url_sequential(url)
        results.append(result)

    return results


# VERSION 2: CONCURRENT
async def fetch_url_async(url: str) -> str:
    print(f"Starting: {url}")
    await asyncio.sleep(1)
    print(f"Finished: {url}")

    return f"Fetched: {url}"


async def run_concurrently() -> list[str]:
    tasks = [
        fetch_url_async(url)
        for url in URLS
    ]

    results = await asyncio.gather(*tasks)

    return results


# COMPARE EXECUTION TIME
def main() -> None:
    print("\n========== SEQUENTIAL ==========")

    sequential_start = time.time()
    sequential_results = run_sequentially()
    sequential_duration = time.time() - sequential_start

    print("\nSequential time:", round(sequential_duration, 2))

    print("\n========== CONCURRENT ==========")

    concurrent_start = time.time()
    concurrent_results = asyncio.run(run_concurrently())
    concurrent_duration = time.time() - concurrent_start

    print("\nConcurrent time:", round(concurrent_duration, 2))

    print("\n========== COMPARISON ==========")

    print(f"Time saved: {sequential_duration - concurrent_duration:.2f} seconds")
    print(
        "Results identical:",
        sequential_results == concurrent_results,
    )


if __name__ == "__main__":
    main()
