"""Evaluate agricultural ASR with a CSV manifest of recorded farmer questions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice import transcribe_audio  # noqa: E402


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref_word in enumerate(reference, 1):
        current = [row]
        for column, hyp_word in enumerate(hypothesis, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (ref_word != hyp_word),
            ))
        previous = current
    return previous[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="CSV with audio_path,reference_text,language columns")
    args = parser.parse_args()
    total_edits = total_reference_words = 0
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for index, row in enumerate(rows, 1):
        path = (args.manifest.parent / row["audio_path"]).resolve()
        result = transcribe_audio(path.read_bytes(), path.name, row.get("language") or "hi")
        reference = words(row["reference_text"])
        hypothesis = words(result["text"])
        edits = edit_distance(reference, hypothesis)
        total_edits += edits
        total_reference_words += len(reference)
        print(f"{index:03d} WER={edits / max(1, len(reference)):.3f} | {result['text']}")
    print(f"Corpus WER={total_edits / max(1, total_reference_words):.3f} ({len(rows)} recordings)")


if __name__ == "__main__":
    main()
