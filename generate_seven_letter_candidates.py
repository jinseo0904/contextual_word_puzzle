#!/usr/bin/env python3
"""
Randomly print 10 candidate words that:
  1. Contain exactly 7 distinct letters (letters may repeat).
  2. Have a wordfreq.word_frequency score above 1e-5.

The candidates are pulled from the shared dictionary JSON that powers the
spelling bee helpers.
"""

import json
import random
from pathlib import Path
from typing import Iterable, List

from wordfreq import word_frequency


DICTIONARY_PATH = Path(
    "/home/mhealth-admin/jin/words_with_friends/words_database/dictionary_compact.json"
)
FREQUENCY_THRESHOLD = 7e-6
NUM_CANDIDATES = 10


def load_words(path: Path) -> List[str]:
    """Return a flat list of words from a JSON list or dict."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [str(word).strip() for word in data]
    if isinstance(data, dict):
        return [str(word).strip() for word in data.keys()]

    raise ValueError("Dictionary JSON must be either a list of words or a dict.")


def has_seven_distinct_letters(word: str) -> bool:
    """True if the word only has letters and exactly 7 unique ones."""
    lowered = word.lower()
    if not lowered.isalpha():
        return False
    return len(set(lowered)) == 7


def filter_candidates(words: Iterable[str]) -> List[str]:
    """Filter to words that satisfy the letter and frequency constraints."""
    candidates: List[str] = []

    for word in words:
        if not has_seven_distinct_letters(word):
            continue

        freq = word_frequency(word.lower(), "en")
        if freq > FREQUENCY_THRESHOLD:
            candidates.append(word.lower())

    return candidates


def main() -> None:
    words = load_words(DICTIONARY_PATH)
    candidates = filter_candidates(words)

    if len(candidates) < NUM_CANDIDATES:
        raise RuntimeError(
            f"Only found {len(candidates)} candidates; need {NUM_CANDIDATES}."
        )

    chosen = random.sample(candidates, NUM_CANDIDATES)

    print(f"Random {NUM_CANDIDATES} candidates (freq > {FREQUENCY_THRESHOLD}):")
    for word in chosen:
        freq = word_frequency(word, "en")
        print(f"- {word} (freq: {freq:.6f})")

    final_pick = random.choice(chosen)
    print(f"\nRandomly selected candidate: {final_pick}")


if __name__ == "__main__":
    main()
