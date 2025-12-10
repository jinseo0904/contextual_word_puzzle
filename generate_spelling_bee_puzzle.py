#!/usr/bin/env python3
"""
Script to generate a Spelling Bee puzzle from candidate words.
Takes a JSON file with candidate words, randomly picks one, selects a center letter,
and finds all valid words using functions from index_mask_dictionary.py
"""

import argparse
import json
import os
import random
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
        word_to_mask
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


def pick_candidate_and_center(candidates: List[Dict[str, Any]], mask_index: Dict[int, List[str]], min_words_required: int = MIN_WORDS_REQUIRED) -> Optional[Tuple[Dict[str, Any], str, List[str]]]:
    """
    Randomly pick a candidate word and try to find a center letter that yields >= MIN_WORDS_REQUIRED words.
    
    Returns:
        Tuple of (candidate_dict, center_letter, valid_words) or None if no valid combination found
    """
    # Shuffle candidates to randomize selection
    candidates_copy = candidates.copy()
    random.shuffle(candidates_copy)
    
    # Try each candidate
    for candidate in candidates_copy:
        word = candidate['word'].lower().strip()
        distinct_letters = candidate['distinct_letters']
        
        # Ensure distinct_letters is a list
        if isinstance(distinct_letters, str):
            # Handle string like "['w', 'a', 'l', 'k', 'i', 'n', 'g']"
            import re
            distinct_letters = [letter.lower().strip().strip("'\"") 
                               for letter in re.findall(r"[a-zA-Z]", distinct_letters)]
        
        distinct_letters = [str(letter).lower().strip() for letter in distinct_letters]
        
        # Validate we have exactly 7 distinct letters
        if len(set(distinct_letters)) != 7:
            print(f"Warning: Candidate '{word}' has {len(set(distinct_letters))} distinct letters, expected 7. Skipping.")
            continue
        
        # Create seed string from distinct letters
        seed = ''.join(sorted(set(distinct_letters)))
        
        # Shuffle distinct letters to randomize center letter selection
        distinct_letters_shuffled = distinct_letters.copy()
        random.shuffle(distinct_letters_shuffled)
        
        # Try each letter as center letter
        for center_letter in distinct_letters_shuffled:
            try:
                # Find all words that can be made from these letters with the center letter required
                valid_words = all_words_for_seed(
                    seed=seed,
                    center_letter=center_letter,
                    mask_index=mask_index,
                    min_len=MIN_WORD_LENGTH
                )
                
                if len(valid_words) >= min_words_required:
                    print(f"✓ Found {len(valid_words)} words for candidate '{word}' with center letter '{center_letter}'")
                    return (candidate, center_letter, valid_words)
                else:
                    print(f"  Tried '{word}' with center '{center_letter}': only {len(valid_words)} words (need {min_words_required})")
            except Exception as e:
                print(f"  Error trying '{word}' with center '{center_letter}': {e}")
                continue
    
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
    
    # Get all distinct letters from candidate
    distinct_letters = candidate['distinct_letters']
    if isinstance(distinct_letters, str):
        import re
        distinct_letters = [letter.lower().strip().strip("'\"") 
                           for letter in re.findall(r"[a-zA-Z]", distinct_letters)]
    distinct_letters = [str(letter).lower().strip() for letter in distinct_letters]
    
    result = {
        "seed_word": candidate['word'],
        "seed_word_clue": candidate.get('clue', ''),
        "distinct_letters": distinct_letters,
        "center_letter": center_letter,
        "total_words": len(word_entries),
        "words": word_entries
    }
    
    return result


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
        help='Path to JSON file with candidate words'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducible results'
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
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    print("=" * 80)
    print("Spelling Bee Puzzle Generator")
    print("=" * 80)
    print()
    
    # Load candidate words
    print(f"Loading candidate words from: {args.candidates_json}")
    data = load_candidate_words(args.candidates_json)
    candidates = data['candidates']
    print(f"✓ Loaded {len(candidates)} candidate(s)")
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


if __name__ == "__main__":
    main()
