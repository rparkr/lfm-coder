# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "httpx",
# ]
# ///

import time

import httpx

# Configuration
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL = "qwen3:0.6b"
# MODEL = "lfm2.5-instruct:latest"
ITERATIONS = 8
PROMPT = "Explain the concept of entropy in exactly two paragraphs."


def main():
    print(f"🔄 Running {ITERATIONS} sequential requests...")

    total_tokens = 0
    start_total = time.perf_counter()

    with httpx.Client() as client:
        for i in range(ITERATIONS):
            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": PROMPT}],
                "stream": False,
            }

            start_req = time.perf_counter()
            response = client.post(OLLAMA_URL, json=payload, timeout=60.0)
            end_req = time.perf_counter()

            if response.status_code == 200:
                tokens = response.json().get("usage", {}).get("completion_tokens", 0)
                total_tokens += tokens
                duration = end_req - start_req
                print(f"✅ Req {i} | Tokens: {tokens:3} | Time: {duration:5.2f}s")
            else:
                print(f"❌ Req {i} failed")

    end_total = time.perf_counter()
    total_duration = end_total - start_total

    print("-" * 45)
    print("🏁 SEQUENTIAL SUMMARY")
    print(f"Total Time:   {total_duration:.2f}s")
    print(f"Average Time: {total_duration / ITERATIONS:.2f}s/req")


if __name__ == "__main__":
    main()
