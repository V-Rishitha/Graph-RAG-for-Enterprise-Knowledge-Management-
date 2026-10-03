"""
Step 4: Collect raw docs from Kubernetes (.md) and Airflow (.rst)
and save them as plain .txt files in a single raw_corpus/ folder.

HOW TO RUN:
1. Put this script in the SAME folder where you ran `git clone`
   (i.e. the folder that contains both `website` and `airflow`).
2. Install one small dependency once:
       pip install markdown docutils beautifulsoup4
3. Run:
       python build_corpus.py

WHAT IT DOES:
- Walks through the Kubernetes docs folders (concepts, tasks, tutorials)
- Walks through the Airflow docs folder (airflow-core/docs)
- Strips Markdown/RST formatting down to plain readable text
- Saves each file as a .txt file inside raw_corpus/
  (kubernetes files get a "k8s_" prefix, airflow files get "af_" prefix)

After this runs, you'll have one folder full of clean .txt files —
that's the input for entity extraction (the next step).
"""

import os
from pathlib import Path

import markdown
from bs4 import BeautifulSoup
from docutils.core import publish_string

# ---- CONFIG: adjust these if your folder names are different ----
K8S_SOURCE_DIRS = [
    "website/content/en/docs/concepts",
    "website/content/en/docs/tasks",
    "website/content/en/docs/tutorials",
]
AIRFLOW_SOURCE_DIR = "airflow/airflow-core/docs"
OUTPUT_DIR = "raw_corpus"
# -------------------------------------------------------------------


def markdown_to_text(md_content: str) -> str:
    """Convert Markdown content to plain text."""
    html = markdown.markdown(md_content)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")


def rst_to_text(rst_content: str) -> str:
    """Convert reStructuredText content to plain text."""
    try:
        html = publish_string(
            source=rst_content,
            writer_name="html",
            settings_overrides={"report_level": 5, "halt_level": 5},
        ).decode("utf-8")
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n")
    except Exception:
        # If RST parsing fails on a weird file, fall back to raw text
        return rst_content


def clean_text(text: str) -> str:
    """Remove excess blank lines and whitespace."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def process_folder(source_dir: str, extension: str, prefix: str, converter, out_dir: Path):
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"  [skip] {source_dir} not found")
        return 0

    count = 0
    for file_path in source_path.rglob(f"*{extension}"):
        try:
            raw = file_path.read_text(encoding="utf-8", errors="ignore")
            text = converter(raw)
            text = clean_text(text)

            if len(text) < 50:
                continue  # skip near-empty files

            # Build a unique, readable output filename
            relative = file_path.relative_to(source_path).with_suffix("")
            safe_name = str(relative).replace(os.sep, "_")
            out_file = out_dir / f"{prefix}_{safe_name}.txt"

            out_file.write_text(text, encoding="utf-8")
            count += 1
        except Exception as e:
            print(f"  [error] {file_path}: {e}")

    return count


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)

    print("Processing Kubernetes docs...")
    k8s_total = 0
    for src in K8S_SOURCE_DIRS:
        k8s_total += process_folder(src, ".md", "k8s", markdown_to_text, out_dir)

    print("Processing Airflow docs...")
    af_total = process_folder(AIRFLOW_SOURCE_DIR, ".rst", "af", rst_to_text, out_dir)

    print(f"\nDone.")
    print(f"  Kubernetes files converted: {k8s_total}")
    print(f"  Airflow files converted:    {af_total}")
    print(f"  Total files in {OUTPUT_DIR}/: {k8s_total + af_total}")


if __name__ == "__main__":
    main()
