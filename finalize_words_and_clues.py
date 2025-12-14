#!/usr/bin/env python3
"""
Post-process validated contextual clues:
  * Deduplicate entries (keep the first occurrence of each word).
  * Retain only clues with score >= threshold (default 4).
  * Merge contextual clues + generic fallback clues with the full pruned word list.
  * Output a JSON object that includes the puzzle metadata (letters + center letter)
    and the list of words, each with: word, contextual_clue, generic_clue, frequency, score.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def deduplicate_clues(clues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for entry in clues:
        word = str(entry.get("word", "")).strip()
        if not word:
            continue
        key = word.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def build_contextual_map(clues: List[Dict[str, Any]], score_threshold: float) -> Dict[str, Dict[str, Any]]:
    ctx_map: Dict[str, Dict[str, Any]] = {}
    for entry in clues:
        word = str(entry.get("word", "")).strip()
        if not word:
            continue
        score = entry.get("score", 0)
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            continue
        if score_value < score_threshold:
            continue
        clue_text = str(entry.get("clue", "")).strip()
        if not clue_text:
            continue
        ctx_map[word.lower()] = {
            "contextual_clue": clue_text,
            "score": score_value
        }
    return ctx_map


def deduplicate_generic_clues(generic_clues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for entry in generic_clues:
        word = str(entry.get("word", "")).strip()
        clue = str(entry.get("clue", "")).strip()
        if not word or not clue:
            continue
        key = word.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append({"word": word, "clue": clue})
    return unique


def build_generic_map(generic_clues: List[Dict[str, Any]]) -> Dict[str, str]:
    g_map: Dict[str, str] = {}
    for entry in generic_clues:
        word = entry["word"].lower()
        g_map[word] = entry["clue"]
    return g_map


def merge_words_with_clues(
    words: List[Dict[str, Any]],
    ctx_map: Dict[str, Dict[str, Any]],
    generic_map: Dict[str, str]
) -> List[Dict[str, Any]]:
    merged = []
    for word_info in words:
        word = str(word_info.get("word", "")).strip()
        if not word:
            continue

        lower = word.lower()
        contextual_data = ctx_map.get(lower)
        if contextual_data:
            contextual_clue = contextual_data["contextual_clue"]
            score = contextual_data["score"]
        else:
            contextual_clue = "N/A"
            score = 0
        generic_clue = generic_map.get(lower, "N/A")

        merged.append({
            "word": word,
            "contextual_clue": contextual_clue,
            "generic_clue": generic_clue,
            "frequency": word_info.get("frequency", 0),
            "score": score
        })
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize spelling bee words and clues.")
    parser.add_argument("--pruned-words", required=True, help="Path to pruned words JSON (with 'words' array).")
    parser.add_argument("--validated-clues", required=True, help="Path to validated contextual clues JSON.")
    parser.add_argument("--generic-clues", required=True, help="Path to JSON with generic clues.")
    parser.add_argument("--output", required=True, help="Path to save the finalized merged JSON.")
    parser.add_argument("--score-threshold", type=float, default=4.0, help="Minimum score required to keep a contextual clue.")
    args = parser.parse_args()

    pruned_path = Path(args.pruned_words)
    validated_path = Path(args.validated_clues)

    generic_path = Path(args.generic_clues)
    output_path = Path(args.output)

    pruned_data = load_json(pruned_path)
    if isinstance(pruned_data, dict) and "words" in pruned_data:
        words = pruned_data["words"]
        metadata = {
            "seed_word": pruned_data.get("seed_word", ""),
            "seed_word_clue": pruned_data.get("seed_word_clue", ""),
            "distinct_letters": pruned_data.get("distinct_letters", []),
            "center_letter": pruned_data.get("center_letter", "")
        }
    elif isinstance(pruned_data, list):
        words = pruned_data
        metadata = {
            "seed_word": "",
            "seed_word_clue": "",
            "distinct_letters": [],
            "center_letter": ""
        }
    else:
        raise ValueError("Pruned words JSON must be a dict with a 'words' key or a list of word entries.")

    validated_data = load_json(validated_path)
    if not isinstance(validated_data, list):
        raise ValueError("Validated clues JSON must be a list.")

    generic_data = load_json(generic_path)
    if not isinstance(generic_data, list):
        raise ValueError("Generic clues JSON must be a list.")

    unique_clues = deduplicate_clues(validated_data)
    contextual_map = build_contextual_map(unique_clues, args.score_threshold)
    unique_generic = deduplicate_generic_clues(generic_data)
    generic_map = build_generic_map(unique_generic)
    merged_entries = merge_words_with_clues(words, contextual_map, generic_map)

    output_data = {
        "seed_word": metadata["seed_word"],
        "seed_word_clue": metadata["seed_word_clue"],
        "distinct_letters": metadata["distinct_letters"],
        "center_letter": metadata["center_letter"],
        "total_words": len(merged_entries),
        "words": merged_entries
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(merged_entries)} finalized entries to {output_path}")


if __name__ == "__main__":
    main()
