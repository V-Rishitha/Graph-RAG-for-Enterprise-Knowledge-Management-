"""
Step 7: Clean up entities.json into a filtered, normalized list of
real technical entities, saved to clean_entities.json.

HOW TO RUN:
    python clean_entities.py

WHAT IT DOES:
1. Loads entities.json (from extract_entities.py)
2. Removes junk: pronouns, stopwords, numbers, single letters
3. Strips leading articles ("a Pod" -> "Pod", "the Node" -> "Node")
4. Normalizes casing and simple plurals so "Pod", "pods", "Pods"
   all become one entity: "Pod"
5. Keeps only entities that appear at least MIN_FREQUENCY times
   across the whole corpus (filters out one-off noise)
6. Saves the final clean list, sorted by frequency, to clean_entities.json
"""

import json
import re
from collections import Counter

import spacy

INPUT_FILE = "entities.json"
OUTPUT_FILE = "clean_entities.json"
MIN_FREQUENCY = 3  # an entity must appear at least this many times to be kept

# Load spaCy just to reuse its stopword list (fast, no need to reprocess text)
nlp = spacy.load("en_core_web_sm")
STOPWORDS = nlp.Defaults.stop_words

# Words that are technically not stopwords but are still useless as entities
EXTRA_JUNK = {
    "example", "note", "default", "true", "false", "name", "type",
    "value", "way", "case", "time", "part", "kind", "thing", "something",
}

LEADING_ARTICLES = re.compile(r"^(a|an|the|this|that|these|those|your|my|our)\s+", re.IGNORECASE)


def normalize(term: str) -> str | None:
    """Clean a single entity/noun-chunk string. Returns None if it should be dropped."""
    term = term.strip()

    # Strip leading articles like "a Pod" -> "Pod"
    term = LEADING_ARTICLES.sub("", term).strip()

    if not term:
        return None

    # Drop pure numbers, pure punctuation, or single characters
    if len(term) <= 1 or term.replace(".", "").replace("-", "").isdigit():
        return None

    # Drop if it's just a stopword or junk word (case-insensitive)
    lowered = term.lower()
    if lowered in STOPWORDS or lowered in EXTRA_JUNK:
        return None

    # Drop if every word in a multi-word term is a stopword (e.g. "of the")
    words = lowered.split()
    if all(w in STOPWORDS for w in words):
        return None

    # Simple plural normalization: "Pods" -> "Pod", "nodes" -> "node"
    # (naive but good enough for a first pass; skip short words to avoid
    # mangling things like "OS" -> "O")
    if len(term) > 3 and term.lower().endswith("s") and not term.lower().endswith("ss"):
        singular = term[:-1]
    else:
        singular = term

    # Title-case single technical words for consistency (Pod, Node, API stay as-is if already caps)
    if singular.isupper() or singular[0].isupper():
        normalized = singular  # keep existing capitalization for proper nouns/acronyms
    else:
        normalized = singular

    return normalized.strip()


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    counts = Counter()
    entity_files = {}  # track which files each entity appears in
    cap_counts = Counter()   # how many times this term appeared capitalized
    total_counts = Counter()  # how many times this term appeared at all (any case)

    for filename, content in data.items():
        terms = [e["text"] for e in content["named_entities"]] + content["noun_chunks"]
        for raw_term in terms:
            normalized = normalize(raw_term)
            if normalized:
                counts[normalized] += 1
                entity_files.setdefault(normalized, set()).add(filename)

            # Track capitalization on the raw (pre-normalize) first word,
            # keyed by lowercase, to decide later if this is a real
            # technical term or just a generic word.
            stripped = LEADING_ARTICLES.sub("", raw_term.strip()).strip()
            if stripped:
                key = stripped.lower()
                total_counts[key] += 1
                first_char = stripped[0]
                if first_char.isupper():
                    cap_counts[key] += 1

    # Merge case-insensitive duplicates (e.g. "pod" and "Pod")
    merged = Counter()
    merged_files = {}
    canonical_form = {}  # lowercase -> the most common actual casing

    for term, count in counts.items():
        key = term.lower()
        merged[key] += count
        merged_files.setdefault(key, set()).update(entity_files[term])
        # Keep whichever casing appeared with higher individual count as canonical
        if key not in canonical_form or count > counts[canonical_form[key]]:
            canonical_form[key] = term

    final_entities = []
    for key, count in merged.items():
        if count < MIN_FREQUENCY:
            continue

        # Require this term to be capitalized most of the time it appears.
        # Real technical terms (Pod, DAG, ConfigMap) are capitalized
        # consistently in docs; generic English nouns (policy, value,
        # argument) mostly aren't. Short acronyms (<=5 chars, all caps
        # somewhere) are always allowed through.
        cap_ratio = cap_counts[key] / total_counts[key] if total_counts[key] else 0
        is_likely_acronym = len(key) <= 5 and canonical_form[key].isupper()
        if cap_ratio < 0.6 and not is_likely_acronym:
            continue

        final_entities.append({
            "entity": canonical_form[key],
            "frequency": count,
            "appears_in_files": len(merged_files[key]),
        })

    final_entities.sort(key=lambda x: x["frequency"], reverse=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_entities, f, indent=2)

    print(f"Kept {len(final_entities)} entities (appearing {MIN_FREQUENCY}+ times).")
    print(f"Saved to {OUTPUT_FILE}\n")

    print("Top 40 cleaned entities:")
    for e in final_entities[:40]:
        print(f"  {e['frequency']:4d}  {e['entity']}")


if __name__ == "__main__":
    main()