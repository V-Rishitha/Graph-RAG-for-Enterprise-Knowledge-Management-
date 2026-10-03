"""
Step 10c: Extract relationships using Groq's free, fast cloud API
(instead of a slow local Ollama model) -- same idea, much faster
hardware, so the full corpus should finish in minutes instead of hours.

PREREQUISITE:
    1. pip install requests
    2. Get a free API key from https://console.groq.com
    3. Set it as an environment variable before running this script:
         Windows PowerShell:  $env:GROQ_API_KEY="your-key-here"
       (Must be set in the SAME terminal window you run this script from.)

HOW TO RUN:
    python extract_relationships_llm.py

WHAT IT DOES:
- Loads your known entities (clean_entities.json)
- Splits each doc into smaller chunks (LLMs work better on focused text)
- For each chunk, asks the LLM: "extract relationships between these
  known entities, using ONLY entities from this list"
- Parses the LLM's JSON response into triples
- Saves everything to relationships_llm.json, after EVERY file (so you
  can safely stop and resume -- it picks up where it left off)
"""

import json
import os
import re
import time
from pathlib import Path

import requests

CORPUS_DIR = "raw_corpus"
ENTITIES_FILE = "clean_entities.json"
OUTPUT_FILE = "relationships_llm.json"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"  # fast Groq-hosted model, free tier, available on this account

CHUNK_SIZE = 3000       # characters per chunk sent to the LLM
LIMIT_FILES = None      # set to None to process everything; start small to test!

if not GROQ_API_KEY:
    raise SystemExit(
        "GROQ_API_KEY environment variable is not set.\n"
        "Run this first in your terminal, then re-run this script:\n"
        '  $env:GROQ_API_KEY="your-key-here"'
    )

with open(ENTITIES_FILE, "r", encoding="utf-8") as f:
    entity_data = json.load(f)
KNOWN_ENTITIES = sorted({e["entity"] for e in entity_data}, key=lambda x: -len(x))
ENTITY_SET_LOWER = {e.lower() for e in KNOWN_ENTITIES}

PROMPT_TEMPLATE = """You are extracting a knowledge graph from technical documentation.

Below is a list of KNOWN ENTITIES (only use these exact terms as subject/object):
{entity_list}

Below is a TEXT PASSAGE. Extract relationships ONLY between entities from the list above
that are actually mentioned in this passage. Use a short relation phrase (2-4 words,
lowercase, like "runs on", "schedules", "is part of", "requires").

Return ONLY a JSON array, nothing else, in this exact format:
[{{"subject": "EntityA", "relation": "relation phrase", "object": "EntityB"}}]

If no valid relationships are found, return an empty array: []

TEXT PASSAGE:
{passage}

JSON:"""


def chunk_text(text: str, size: int):
    """Split text into chunks, breaking at paragraph boundaries where possible."""
    paragraphs = text.split("\n")
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) > size and current:
            chunks.append(current)
            current = p
        else:
            current += "\n" + p
    if current.strip():
        chunks.append(current)
    return chunks


def entities_in_chunk(chunk: str):
    """Find which known entities actually appear in this chunk (case-insensitive)."""
    chunk_lower = chunk.lower()
    found = [e for e in KNOWN_ENTITIES if e.lower() in chunk_lower]
    return found[:60]  # cap the list length so the prompt doesn't get huge


def call_ollama(prompt: str) -> str:
    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 500,
        },
        timeout=60,
    )
    if response.status_code == 429:
        # Free tier rate limit hit -- wait a bit and retry once
        time.sleep(10)
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 500,
            },
            timeout=60,
        )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def parse_triples(llm_output: str):
    """Extract a JSON array from the LLM's response, tolerating extra text around it."""
    match = re.search(r"\[.*\]", llm_output, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    valid = []
    for item in data:
        if not isinstance(item, dict):
            continue
        subj, rel, obj = item.get("subject"), item.get("relation"), item.get("object")
        if not (subj and rel and obj):
            continue
        # Only keep it if subject/object are actually known entities
        if subj.lower() in ENTITY_SET_LOWER and obj.lower() in ENTITY_SET_LOWER:
            valid.append({"subject": subj, "relation": rel.lower(), "object": obj})
    return valid


def process_file(file_path: Path):
    text = file_path.read_text(encoding="utf-8")
    chunks = chunk_text(text, CHUNK_SIZE)

    file_triples = []
    for chunk in chunks:
        candidates = entities_in_chunk(chunk)
        if len(candidates) < 2:
            continue  # need at least 2 entities present to find a relationship

        prompt = PROMPT_TEMPLATE.format(
            entity_list=", ".join(candidates),
            passage=chunk[:CHUNK_SIZE],
        )
        try:
            output = call_ollama(prompt)
            triples = parse_triples(output)
            for t in triples:
                t["source_file"] = file_path.name
            file_triples.extend(triples)
        except Exception as e:
            print(f"    [error on chunk] {e}")

    return file_triples


def main():
    corpus_path = Path(CORPUS_DIR)
    txt_files = sorted(corpus_path.glob("*.txt"))
    if LIMIT_FILES:
        txt_files = txt_files[:LIMIT_FILES]

    print(f"Processing {len(txt_files)} files with {MODEL} via Ollama...")
    print("(This is slower than spaCy -- progress will print per file)")
    print("Progress is saved after every file, so you can safely stop (Ctrl+C) and resume later.\n")

    # RESUME SUPPORT: if an output file already exists, load it and skip
    # any files we've already processed (tracked in a .progress file).
    progress_file = OUTPUT_FILE + ".progress"
    all_triples = []
    done_files = set()
    if Path(OUTPUT_FILE).exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            all_triples = json.load(f)
    if Path(progress_file).exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            done_files = set(json.load(f))
        print(f"Resuming: {len(done_files)} files already done, {len(all_triples)} triples so far.\n")

    for i, file_path in enumerate(txt_files, 1):
        if file_path.name in done_files:
            continue  # already processed in a previous run

        triples = process_file(file_path)
        all_triples.extend(triples)
        done_files.add(file_path.name)
        print(f"  [{i}/{len(txt_files)}] {file_path.name}: {len(triples)} triples (total: {len(all_triples)})")

        # Save progress after EVERY file so nothing is lost if interrupted
        seen = {}
        for t in all_triples:
            key = (t["subject"].lower(), t["relation"], t["object"].lower())
            seen[key] = t
        deduped_so_far = list(seen.values())
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(deduped_so_far, f, indent=2)
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(sorted(done_files), f)

    # Final dedupe (same as the incremental one, just for the summary print)
    seen = {}
    for t in all_triples:
        key = (t["subject"].lower(), t["relation"], t["object"].lower())
        seen[key] = t
    deduped = list(seen.values())

    print(f"\nDone. {len(all_triples)} raw triples, {len(deduped)} unique.")
    print(f"Saved to {OUTPUT_FILE}")
    print("\nSample:")
    for t in deduped[:15]:
        print(f"  ({t['subject']}) --[{t['relation']}]--> ({t['object']})")


if __name__ == "__main__":
    main()