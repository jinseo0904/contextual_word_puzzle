#!/usr/bin/env python3
"""
Script to generate a Spelling Bee puzzle from candidate words.
Takes a JSON file with candidate words, randomly picks one, selects a center letter,
and finds all valid words using functions from index_mask_dictionary.py
"""

import argparse
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add words_database directory to path to import index_mask_dictionary
_words_db_path = Path(__file__).parent.parent / "words_database"
if str(_words_db_path) not in sys.path:
    sys.path.insert(0, str(_words_db_path))

try:
    from index_mask_dictionary import (
        load_words_from_json,
        build_mask_index,
        all_words_for_seed,
    )
    from wordfreq import word_frequency
except ImportError as e:
    print(f"Error importing required modules: {e}", file=sys.stderr)
    print(f"Tried to import from: {_words_db_path}", file=sys.stderr)
    print("Make sure index_mask_dictionary.py is in the words_database directory", file=sys.stderr)
    sys.exit(1)


# Configuration
DICTIONARY_PATH = Path("/home/mhealth-admin/jin/words_with_friends/words_database/dictionary_compact.json")
MIN_WORD_LENGTH = 4
MIN_WORDS_REQUIRED = 20
SUBSET_FREQ_THRESHOLD = 7e-6
OUTPUT_DIR = Path("/home/mhealth-admin/jin/words_with_friends/spelling_bee/generated_jsons")


def load_candidate_words(json_path: str) -> Dict[str, Any]:
    """Load candidate words from JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'candidates' not in data:
            raise ValueError("JSON file must contain a 'candidates' field")
        
        candidates = data['candidates']
        if not isinstance(candidates, list) or len(candidates) == 0:
            raise ValueError("'candidates' must be a non-empty list")
        
        return data
    except FileNotFoundError:
        print(f"Error: File not found: {json_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading candidate words: {e}", file=sys.stderr)
        sys.exit(1)


def parse_distinct_letters(distinct_letters: Any) -> List[str]:
    """Normalize distinct letters into a list of lowercase single characters."""
    if isinstance(distinct_letters, list):
        letters = [str(letter).lower().strip() for letter in distinct_letters]
    elif isinstance(distinct_letters, str):
        letters = [letter.lower() for letter in re.findall(r"[a-zA-Z]", distinct_letters)]
    else:
        letters = []
    return [letter for letter in letters if letter]


def evaluate_candidate(
    candidate: Dict[str, Any],
    mask_index: Dict[int, List[str]],
    min_words_required: int,
    verbose: bool = True
) -> Optional[Tuple[Dict[str, Any], str, List[str]]]:
    """
    Try all center letters for a candidate and return the first combination
    that meets the min_words_required threshold.
    """
    word = str(candidate.get('word', '')).lower().strip()
    distinct_letters = parse_distinct_letters(candidate.get('distinct_letters', []))
    unique_letters = sorted(set(distinct_letters))

    if len(unique_letters) != 7:
        if verbose:
            print(f"Warning: Candidate '{word}' has {len(unique_letters)} distinct letters, expected 7. Skipping.")
        return None

    seed = ''.join(unique_letters)

    centers = unique_letters.copy()
    random.shuffle(centers)

    for center_letter in centers:
        try:
            valid_words = all_words_for_seed(
                seed=seed,
                center_letter=center_letter,
                mask_index=mask_index,
                min_len=MIN_WORD_LENGTH
            )
        except Exception as e:
            if verbose:
                print(f"  Error trying '{word}' with center '{center_letter}': {e}")
            continue

        if len(valid_words) >= min_words_required:
            if verbose:
                print(f"✓ Found {len(valid_words)} words for candidate '{word}' with center letter '{center_letter}'")
            return candidate, center_letter, valid_words

        if verbose:
            print(f"  Tried '{word}' with center '{center_letter}': only {len(valid_words)} words (need {min_words_required})")

    return None


def pick_candidate_and_center(
    candidates: List[Dict[str, Any]],
    mask_index: Dict[int, List[str]],
    min_words_required: int = MIN_WORDS_REQUIRED
) -> Optional[Tuple[Dict[str, Any], str, List[str]]]:
    """
    Randomly pick a candidate word and try to find a center letter that yields >= MIN_WORDS_REQUIRED words.
    
    Returns:
        Tuple of (candidate_dict, center_letter, valid_words) or None if no valid combination found
    """
    candidates_copy = candidates.copy()
    random.shuffle(candidates_copy)

    for candidate in candidates_copy:
        result = evaluate_candidate(candidate, mask_index, min_words_required, verbose=True)
        if result:
            return result

    return None


def get_word_definition(word: str, dict_data: Dict[str, Any]) -> str:
    """Get definition for a word from dictionary data."""
    # Try exact match (lowercase)
    word_lower = word.lower()
    if word_lower in dict_data:
        return dict_data[word_lower]
    
    # Try case-insensitive match
    for key, value in dict_data.items():
        if key.lower() == word_lower:
            return value
    
    return "No definition available"


def generate_puzzle_json(
    candidate: Dict[str, Any],
    center_letter: str,
    valid_words: List[str],
    dict_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate the final puzzle JSON with words sorted by frequency (high to low).
    
    Args:
        candidate: The selected candidate word dict
        center_letter: The selected center letter
        valid_words: List of all valid words
        dict_data: Dictionary data for definitions
    
    Returns:
        Dictionary with puzzle data
    """
    # Get frequency for each word and sort by frequency (descending)
    words_with_freq = []
    for word in valid_words:
        try:
            freq = word_frequency(word, 'en')
        except Exception:
            freq = 0.0
        
        words_with_freq.append((word, freq))
    
    # Sort by frequency (descending), then alphabetically
    words_with_freq.sort(key=lambda x: (-x[1], x[0]))
    
    # Build result with definitions
    word_entries = []
    for word, freq in words_with_freq:
        definition = get_word_definition(word, dict_data)
        word_entries.append({
            "word": word,
            "frequency": freq,
            "definition": definition
        })

    word_entries = drop_superset_words(word_entries, SUBSET_FREQ_THRESHOLD)
    
    # Get all distinct letters from candidate
    distinct_letters = parse_distinct_letters(candidate.get('distinct_letters', []))
    
    result = {
        "seed_word": candidate['word'],
        "seed_word_clue": candidate.get('clue', ''),
        "distinct_letters": distinct_letters,
        "center_letter": center_letter,
        "total_words": len(word_entries),
        "words": word_entries
    }
    
    return result


def drop_superset_words(
    word_entries: List[Dict[str, Any]],
    freq_threshold: float
) -> List[Dict[str, Any]]:
    """
    Remove any word that strictly contains another retained word whose frequency
    meets or exceeds freq_threshold.
    Prints the dropping process for debugging.
    """
    if not word_entries:
        return word_entries

    sorted_by_len = sorted(
        word_entries, key=lambda entry: (len(entry["word"]), entry["word"])
    )
    kept_entries: List[Dict[str, Any]] = []
    dropped: List[Tuple[Dict[str, Any], Dict[str, Any], str]] = []

    for entry in sorted_by_len:
        word = entry["word"]
        drop_reason = next(
            (
                kept_entry
                for kept_entry in kept_entries
                if kept_entry["word"] in word
                and kept_entry["word"] != word
                and kept_entry["frequency"] >= freq_threshold
            ),
            None,
        )
        if drop_reason is not None:
            if is_simple_affix_variant(drop_reason["word"], word):
                dropped.append((entry, drop_reason, "affix"))
                continue

            if entry["frequency"] <= freq_threshold:
                dropped.append((entry, drop_reason, "frequency"))
                continue

        kept_entries.append(entry)

    if dropped:
        print("Dropping superset words due to containment:")
        for superset_entry, subset_entry, reason in dropped:
            sup_freq = superset_entry["frequency"]
            sub_freq = subset_entry["frequency"]
            print(
                f"  - Dropped '{superset_entry['word']}' (freq: {sup_freq:.2e}) "
                f"because it contains '{subset_entry['word']}' (freq: {sub_freq:.2e})"
                f" [{reason}]"
            )
        print()

    kept_set = set(entry["word"] for entry in kept_entries)
    return [entry for entry in word_entries if entry["word"] in kept_set]


def is_simple_affix_variant(base: str, candidate: str) -> bool:
    """Return True if candidate is base with a simple affix like -ing, -er, -s, -ness, or re-."""
    base = base.lower()
    candidate = candidate.lower()
    if not base or not candidate:
        return False

    # Prefix check
    simple_prefixes = ("re",)
    for prefix in simple_prefixes:
        if candidate.startswith(prefix) and candidate[len(prefix):] == base:
            return True

    # Suffix checks
    suffixes = ("ing", "er", "ness", "s")
    for suffix in suffixes:
        if not candidate.endswith(suffix):
            continue

        stripped = candidate[:-len(suffix)] if suffix else candidate
        if stripped == base:
            return True

        if suffix in ("ing", "er") and len(base) >= 1 and stripped == base + base[-1]:
            return True

    return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Generate Spelling Bee puzzle from candidate words JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_spelling_bee_puzzle.py candidates.json
  python generate_spelling_bee_puzzle.py candidates.json --seed 42
  python generate_spelling_bee_puzzle.py candidates.json --output puzzle.json
        """
    )
    parser.add_argument(
        'candidates_json',
        type=str,
        nargs='?',
        default=None,
        help='Path to JSON file with candidate words (required unless --seed-word is used)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducible results'
    )
    parser.add_argument(
        '--seed-word',
        type=str,
        default=None,
        help='Specify a particular candidate word from the JSON to force as the seed'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output filename (default: auto-generated with timestamp)'
    )
    parser.add_argument(
        '--dictionary',
        type=str,
        default=str(DICTIONARY_PATH),
        help=f'Path to dictionary file (default: {DICTIONARY_PATH})'
    )
    parser.add_argument(
        '--min-words',
        type=int,
        default=MIN_WORDS_REQUIRED,
        help=f'Minimum number of words required (default: {MIN_WORDS_REQUIRED})'
    )
    
    args = parser.parse_args()

    if args.seed_word is None and args.candidates_json is None:
        print("Error: You must provide a candidates JSON file unless --seed-word is specified.", file=sys.stderr)
        sys.exit(1)
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    print("=" * 80)
    print("Spelling Bee Puzzle Generator")
    print("=" * 80)
    print()
    
    # Load candidate words
    candidates: List[Dict[str, Any]] = []
    if args.candidates_json:
        print(f"Loading candidate words from: {args.candidates_json}")
        data = load_candidate_words(args.candidates_json)
        candidates = data['candidates']
        print(f"✓ Loaded {len(candidates)} candidate(s)")
        print()
    else:
        print("No candidate JSON provided; running in direct seed-word mode.")
        print()
    
    # Load dictionary
    print(f"Loading dictionary from: {args.dictionary}")
    dict_path = Path(args.dictionary)
    if not dict_path.exists():
        print(f"Error: Dictionary file not found: {dict_path}", file=sys.stderr)
        sys.exit(1)
    
    words, dict_data = load_words_from_json(dict_path)
    print(f"✓ Loaded {len(words)} words from dictionary")
    print()
    
    # Build mask index
    print("Building mask index (this may take a moment)...")
    mask_index = build_mask_index(words, min_len=MIN_WORD_LENGTH)
    indexed_count = sum(len(v) for v in mask_index.values())
    print(f"✓ Indexed {indexed_count} words (length >= {MIN_WORD_LENGTH})")
    print()
    
    # Find valid candidate and center letter combination
    if args.seed_word:
        desired_word = args.seed_word.lower().strip()
        candidate = None
        if candidates:
            candidate = next(
                (
                    cand for cand in candidates
                    if str(cand.get('word', '')).lower().strip() == desired_word
                ),
                None
            )
            if candidate is None and args.candidates_json:
                print(f"Warning: Seed word '{args.seed_word}' not found in {args.candidates_json}. Falling back to direct input.")

        if candidate is None:
            filtered_letters = [ch for ch in args.seed_word if ch.isalpha()]
            distinct_letters = list(set(filtered_letters if filtered_letters else list(args.seed_word)))
            distinct_letters.sort()

            assert len(distinct_letters) == 7, f"Seed word '{args.seed_word}' has {len(distinct_letters)} distinct letters, expected 7"
            candidate = {
            "word": args.seed_word.upper(),
            "clue": "",
            "distinct_letters": distinct_letters
        }
        assert candidate is not None, f"Failed to create candidate for seed word '{args.seed_word}'"

        print(f"Attempting to build puzzle using provided seed word: '{desired_word}'")
        result = evaluate_candidate(candidate, mask_index, args.min_words, verbose=True)
        if result is None:
            print(f"\nERROR: Provided seed word '{args.seed_word}' does not produce >= {args.min_words} valid words.", file=sys.stderr)
            print("Consider lowering --min-words or choosing a different seed word.", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Searching for candidate word with >= {args.min_words} valid words...")
        print()
        result = pick_candidate_and_center(candidates, mask_index, min_words_required=args.min_words)
    
    if result is None:
        print("\n" + "=" * 80)
        print("ERROR: Could not find a candidate word with sufficient valid words")
        print("=" * 80)
        print(f"Tried all {len(candidates)} candidates, but none yielded >= {args.min_words} words")
        print("You may need to:")
        print("  1. Lower the --min-words threshold")
        print("  2. Generate new candidate words with better letters")
        sys.exit(1)
    
    candidate, center_letter, valid_words = result
    print()
    
    # Generate puzzle JSON
    print("Generating puzzle JSON with word frequencies and definitions...")
    puzzle_data = generate_puzzle_json(candidate, center_letter, valid_words, dict_data)
    print(f"✓ Generated puzzle with {puzzle_data['total_words']} words")
    print()
    
    # Determine output filename
    if args.output:
        output_filename = args.output
        if not output_filename.endswith('.json'):
            output_filename += '.json'
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        seed_word = candidate['word'].lower()
        output_filename = f"spelling_bee_puzzle_{seed_word}_{timestamp}.json"
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_filename

    # Save puzzle JSON
    print(f"Saving puzzle to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(puzzle_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved puzzle JSON")
    print()

    # Display summary
    print("=" * 80)
    print("PUZZLE SUMMARY:")
    print("=" * 80)
    print(f"Seed word: {puzzle_data['seed_word']}")
    print(f"Clue: {puzzle_data['seed_word_clue']}")
    print(f"Distinct letters: {', '.join(sorted(puzzle_data['distinct_letters']))}")
    print(f"Center letter: {puzzle_data['center_letter']} (required in all words)")
    print(f"Total valid words: {puzzle_data['total_words']}")
    print(f"Words sorted by frequency (high to low)")
    print()
    print("Top 10 words by frequency:")
    for i, entry in enumerate(puzzle_data['words'][:10], 1):
        freq_str = f"{entry['frequency']:.2e}" if entry['frequency'] > 0 else "0.00e+00"
        print(f"  {i:2}. {entry['word']:20} (freq: {freq_str})")
    print()
    print("=" * 80)
    print("Done!")
    print("=" * 80)
    print(f"\nFull path to generated JSON: {output_path.resolve()}\n")


if __name__ == "__main__":
    main()
