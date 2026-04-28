# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "httpx",
# ]
# ///

import asyncio
import time

import httpx

# Configuration
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
# MODEL = "qwen3:0.6b"
# As of 2026-04-23, Ollama does not support parallel execution on the lfm2 architecture
MODEL = "lfm2.5-instruct:latest"
CONCURRENT_REQUESTS = 8
PROMPT = "Explain the concept of entropy in exactly two paragraphs."


async def make_request(client, request_id):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
    }

    start_time = time.perf_counter()
    try:
        response = await client.post(OLLAMA_URL, json=payload, timeout=120.0)
        end_time = time.perf_counter()

        if response.status_code == 200:
            data = response.json()
            # Extract token count from OpenAI-compatible response
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            duration = end_time - start_time
            tps = tokens / duration if duration > 0 else 0

            print(
                f"✅ Req {request_id} | Tokens: {tokens:3} | Time: {duration:5.2f}s | Speed: {tps:5.2f} tok/s"
            )
            return tokens
        else:
            print(f"❌ Req {request_id} failed: {response.status_code}")
            return 0
    except Exception as e:
        print(f"⚠️ Req {request_id} error: {e}")
        return 0


async def main():
    print(f"🚀 Testing {CONCURRENT_REQUESTS} concurrent requests (OpenAI Endpoint)...")
    print(f"📊 Model: {MODEL}\n")

    start_total = time.perf_counter()

    async with httpx.AsyncClient() as client:
        tasks = [make_request(client, i) for i in range(CONCURRENT_REQUESTS)]
        token_counts = await asyncio.gather(*tasks)

    end_total = time.perf_counter()
    total_duration = end_total - start_total
    total_tokens = sum(token_counts)
    aggregate_tps = total_tokens / total_duration if total_duration > 0 else 0

    print("-" * 60)
    print("🏁 SUMMARY")
    print(f"Total Tokens: {total_tokens}")
    print(f"Total Time:   {total_duration:.2f}s")
    print(f"Average Time: {total_duration / CONCURRENT_REQUESTS:.2f}s/req")
    print(f"System Throughput (Aggregate): {aggregate_tps:.2f} tok/s")


if __name__ == "__main__":
    asyncio.run(main())
