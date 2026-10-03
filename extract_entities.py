"""
Step 6: Extract entities from every .txt file in raw_corpus/
using spaCy, and save the results to entities.json for inspection.

HOW TO RUN:
    python extract_entities.py

WHAT IT DOES:
- Loads spaCy's small English model
- Reads every .txt file in raw_corpus/
- For each file, extracts:
    - Named entities spaCy finds automatically (people, orgs, products, etc.)
    - Noun chunks (multi-word technical terms like "persistent volume claim")
- Saves everything into entities.json, organized by source file

WHY BOTH named entities AND noun chunks:
spaCy's built-in NER is trained on general text (news, Wikipedia) so it's
good at organizations/products but WON'T reliably catch technical jargon
like "Pod", "DAG", "Operator". Noun chunks catch those because they just
look for noun phrases, regardless of whether spaCy "knows" what they are.
We'll clean/filter this combined list in the next step.
"""

import json
from pathlib import Path
from collections import Counter

import spacy

CORPUS_DIR = "raw_corpus"
OUTPUT_FILE = "entities.json"

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")


def extract_from_text(text: str):
    """Run spaCy on text and return named entities + noun chunks."""
    # Limit very long docs to keep spaCy fast (spaCy default max length is 1M chars anyway)
    doc = nlp(text[:100000])

    named_entities = [
        {"text": ent.text.strip(), "label": ent.label_}
        for ent in doc.ents
        if len(ent.text.strip()) > 1
    ]

    noun_chunks = [
        chunk.text.strip()
        for chunk in doc.noun_chunks
        if 1 < len(chunk.text.strip()) < 50  # skip 1-char or overly long chunks
    ]

    return named_entities, noun_chunks


def main():
    corpus_path = Path(CORPUS_DIR)
    txt_files = sorted(corpus_path.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {CORPUS_DIR}/. Did you run build_corpus.py first?")
        return

    print(f"Found {len(txt_files)} files. Extracting entities...\n")

    results = {}
    all_entity_texts = Counter()

    for i, file_path in enumerate(txt_files, 1):
        text = file_path.read_text(encoding="utf-8")
        named_entities, noun_chunks = extract_from_text(text)

        results[file_path.name] = {
            "named_entities": named_entities,
            "noun_chunks": noun_chunks,
        }

        for ent in named_entities:
            all_entity_texts[ent["text"]] += 1
        for chunk in noun_chunks:
            all_entity_texts[chunk] += 1

        if i % 50 == 0 or i == len(txt_files):
            print(f"  Processed {i}/{len(txt_files)} files...")

    # Save full results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Full results saved to {OUTPUT_FILE}")

    # Show the most common terms found, as a sanity check
    print("\nTop 30 most frequent terms found across the corpus:")
    for term, count in all_entity_texts.most_common(30):
        print(f"  {count:4d}  {term}")


if __name__ == "__main__":
    main()