"""
Step 8: Extract relationships between entities.

For every sentence in raw_corpus/, if two of our known entities
(from clean_entities.json) appear in the same sentence, use spaCy's
dependency parser to find the connecting verb phrase between them.

HOW TO RUN:
    python extract_relationships.py

OUTPUT:
    relationships.json -- a list of triples:
        {"subject": "Pod", "relation": "runs on", "object": "Node",
         "sentence": "...", "source_file": "..."}

NOTE: This is a simple, rule-based approach (not a trained relation
extraction model). It won't catch everything, and some relations will
be noisy -- that's expected and fine for a first working version. It's
a solid baseline you can improve on later (e.g. compare it against a
transformer-based relation extractor for your paper).
"""

import json
import re
from pathlib import Path

import spacy

CORPUS_DIR = "raw_corpus"
ENTITIES_FILE = "clean_entities.json"
OUTPUT_FILE = "relationships.json"

MAX_FILES = None  # set to a small number like 20 for a quick test run first

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

print("Loading known entities...")
with open(ENTITIES_FILE, "r", encoding="utf-8") as f:
    entity_data = json.load(f)

# Build a fast lookup set of known entity names (lowercase for matching)
KNOWN_ENTITIES = {e["entity"].lower() for e in entity_data}
print(f"  Loaded {len(KNOWN_ENTITIES)} known entities.\n")


def entity_match(token) -> str | None:
    """If this token (or token+next) matches a known entity, return its text."""
    if token.text.lower() in KNOWN_ENTITIES:
        return token.text
    if token.i + 1 < len(token.doc):
        two_word = f"{token.text} {token.doc[token.i + 1].text}".lower()
        if two_word in KNOWN_ENTITIES:
            return f"{token.text} {token.doc[token.i + 1].text}"
    return None


def extract_svo_triples(sent):
    """
    Strict subject-verb-object extraction: for each verb in the sentence,
    find its grammatical subject and object, and only keep the triple if
    BOTH sides are known entities. This is far more reliable than chasing
    ancestor chains -- it's the standard dependency-parsing pattern for
    relation extraction.
    """
    triples = []
    for token in sent:
        if token.pos_ not in ("VERB", "AUX"):
            continue

        # Find subject(s) of this verb
        subjects = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
        if not subjects:
            continue

        # Find object(s): direct object, attribute, or object of an attached preposition
        objects = []
        prep_word = None
        for c in token.children:
            if c.dep_ in ("dobj", "attr", "oprd"):
                objects.append(c)
            elif c.dep_ == "prep":
                prep_word = c.text
                objects.extend([gc for gc in c.children if gc.dep_ == "pobj"])

        if not objects:
            continue

        for subj in subjects:
            subj_entity = entity_match(subj)
            if not subj_entity:
                continue
            for obj in objects:
                obj_entity = entity_match(obj)
                if not obj_entity or obj_entity.lower() == subj_entity.lower():
                    continue

                relation = token.lemma_
                if prep_word:
                    relation = f"{relation} {prep_word}"

                triples.append({
                    "subject": subj_entity,
                    "relation": relation,
                    "object": obj_entity,
                    "sentence": sent.text.strip().replace("\n", " "),
                })

        # PASSIVE VOICE: "Pod is scheduled by scheduler" -- the nsubjpass
        # ("Pod") is actually the one receiving the action, and the true
        # doer is the "by X" agent. So the triple direction flips:
        # (agent) --[verb]--> (nsubjpass).
        passive_subjects = [c for c in token.children if c.dep_ == "nsubjpass"]
        agent_phrase = [c for c in token.children if c.dep_ == "agent"]
        if passive_subjects and agent_phrase:
            agents = [gc for gc in agent_phrase[0].children if gc.dep_ == "pobj"]
            for receiver in passive_subjects:
                receiver_entity = entity_match(receiver)
                if not receiver_entity:
                    continue
                for agent in agents:
                    agent_entity = entity_match(agent)
                    if not agent_entity or agent_entity.lower() == receiver_entity.lower():
                        continue
                    triples.append({
                        "subject": agent_entity,
                        "relation": token.lemma_,
                        "object": receiver_entity,
                        "sentence": sent.text.strip().replace("\n", " "),
                    })

    return triples


def extract_noun_prep_triples(sent):
    """
    Catch noun-to-noun relationships that don't involve a verb at all,
    e.g. "API server for Kubernetes" or "component of the cluster".
    Pattern: NOUN --prep--> NOUN, where the preposition attaches
    directly to a noun (not a verb).
    """
    triples = []
    for token in sent:
        if token.dep_ != "prep" or token.head.pos_ not in ("NOUN", "PROPN"):
            continue

        head_entity = entity_match(token.head)
        if not head_entity:
            continue

        for pobj in token.children:
            if pobj.dep_ != "pobj":
                continue
            obj_entity = entity_match(pobj)
            if not obj_entity or obj_entity.lower() == head_entity.lower():
                continue
            triples.append({
                "subject": head_entity,
                "relation": token.text,
                "object": obj_entity,
                "sentence": sent.text.strip().replace("\n", " "),
            })
    return triples


CODE_LIKE_PATTERN = re.compile(r"[{}=<>|]|^\s*-\s|:\s*$|^\s*#")


def looks_like_code(sent_text: str) -> bool:
    """Heuristic: skip sentences that are really YAML/code/CLI snippets, not prose."""
    stripped = sent_text.strip()
    if not stripped:
        return True
    # Too short to be a real sentence
    if len(stripped.split()) < 4:
        return True
    # Contains code-like symbols
    if CODE_LIKE_PATTERN.search(stripped):
        return True
    # No lowercase letters at all (likely a command or constant block)
    if not any(c.islower() for c in stripped):
        return True
    return False


def process_file(file_path: Path):
    text = file_path.read_text(encoding="utf-8")
    # Keep it manageable per file
    doc = nlp(text[:100000])

    triples = []
    for sent in doc.sents:
        if looks_like_code(sent.text):
            continue

        for t in extract_svo_triples(sent):
            t["source_file"] = file_path.name
            triples.append(t)

        for t in extract_noun_prep_triples(sent):
            t["source_file"] = file_path.name
            triples.append(t)

    return triples


def main():
    corpus_path = Path(CORPUS_DIR)
    txt_files = sorted(corpus_path.glob("*.txt"))
    if MAX_FILES:
        txt_files = txt_files[:MAX_FILES]

    print(f"Processing {len(txt_files)} files for relationships...\n")

    all_triples = []
    for i, file_path in enumerate(txt_files, 1):
        triples = process_file(file_path)
        all_triples.extend(triples)

        if i % 50 == 0 or i == len(txt_files):
            print(f"  Processed {i}/{len(txt_files)} files... ({len(all_triples)} triples so far)")

    # Deduplicate identical (subject, relation, object) triples, keeping
    # one example sentence per unique triple
    seen = {}
    for t in all_triples:
        key = (t["subject"].lower(), t["relation"], t["object"].lower())
        if key not in seen:
            seen[key] = t
    deduped = list(seen.values())

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2)

    print(f"\nExtracted {len(all_triples)} raw triples, {len(deduped)} unique after dedup.")
    print(f"Saved to {OUTPUT_FILE}")
    all_triples = deduped

    print("\nSample of 15 extracted triples:")
    for t in all_triples[:15]:
        print(f"  ({t['subject']}) --[{t['relation']}]--> ({t['object']})")


if __name__ == "__main__":
    main()