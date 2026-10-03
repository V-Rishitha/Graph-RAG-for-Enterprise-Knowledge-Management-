import os
import requests

key = os.environ.get("GROQ_API_KEY")
print("Key found:", bool(key), "| starts with:", key[:8] if key else None)

# Test 1: list available models (simplest possible check)
r = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {key}"},
)
print("\nModels list status:", r.status_code)
import json as _json
data = r.json()
print("\nAll available model IDs:")
for m in data.get("data", []):
    print(" -", m["id"])

# Test 2: a real chat completion call
r2 = requests.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={"Authorization": f"Bearer {key}"},
    json={
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": "Say hi in 3 words."}],
    },
)
print("\nChat completion status:", r2.status_code)
print(r2.text[:1000])