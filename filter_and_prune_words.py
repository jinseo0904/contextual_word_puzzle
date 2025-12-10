#!/usr/bin/env python3
"""
Filter and prune words from a spelling bee puzzle JSON using LLM-based filtering.
Processes words in batches and uses Ollama to filter inappropriate words.
"""

import json
import argparse
import re
import requests
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


# Configuration
MUST_INCLUDE_WORD_FREQUENCY_THRESHOLD = 5e-06
BATCH_SIZE = 5
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gpt-oss:20b"
TIMEOUT = 300  # 5 minutes timeout


def load_prompt_template(prompt_path: str) -> str:
    """Load the pruning prompt template from file."""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Prompt template file not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading prompt template: {e}", file=sys.stderr)
        sys.exit(1)


def format_words_for_prompt(words: List[Dict[str, Any]]) -> str:
    """Format a list of word objects into the prompt format.
    
    Includes both word and definition for each word to help the LLM make decisions.
    """
    lines = []
    for word_obj in words:
        word = word_obj.get("word", "")
        definition = word_obj.get("definition", "")
        
        # Warn if definition is missing
        if not definition:
            definition = "No definition available"
            print(f"    Warning: Word '{word}' has no definition in input JSON")
        
        lines.append(f"{word}: {definition}")
    return "\n".join(lines)


def merge_prompt_with_words(template: str, words_text: str) -> str:
    """Merge the prompt template with the words list."""
    # Look for placeholder <<>>
    if "<<>>" in template:
        return template.replace("<<>>", words_text)
    else:
        # If no placeholder, append words at the end
        return template + "\n\n" + words_text


def extract_json_from_response(response: str) -> Optional[str]:
    """Extract JSON from response, handling markdown code blocks and extra text."""
    # Try to find JSON in markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    # Try to find JSON object directly
    json_match = re.search(r'(\{.*\})', response, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    return None


def query_ollama(prompt: str, model: str = OLLAMA_MODEL, ollama_url: str = OLLAMA_URL) -> str:
    """Send prompt to Ollama and get response."""
    url = f"{ollama_url}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', '')
    
    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"Could not connect to Ollama at {ollama_url}. Make sure Ollama is running.")
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Request to Ollama timed out after {TIMEOUT} seconds.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error querying Ollama: {e}")


def process_batch(
    batch: List[Dict[str, Any]],
    prompt_template: str,
    batch_num: int,
    total_batches: int
) -> Tuple[List[str], Dict[str, str]]:
    """
    Process a batch of words through Ollama.
    
    NOTE: The LLM returns only word strings (not definitions). The full word objects
    with definitions are matched back to these strings in the main function.
    
    Returns:
        Tuple of (kept_word_strings, removed_words_with_explanations)
        - kept_word_strings: List of word strings that should be kept
        - removed_words_with_explanations: Dict mapping word -> explanation
    """
    print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} words)...")
    
    # Format words for prompt - includes both word AND definition for each word
    # Format: "word: definition" for each word in the batch
    words_text = format_words_for_prompt(batch)
    
    # Verify definitions are included (for debugging)
    words_with_defs = sum(1 for w in batch if w.get("definition") and w.get("definition", "").strip())
    words_with_empty_defs = len(batch) - words_with_defs
    if words_with_empty_defs > 0:
        print(f"    Warning: {words_with_empty_defs}/{len(batch)} words have empty/missing definitions")
    else:
        # Confirm definitions are included in the prompt
        print(f"    Sending {len(batch)} words with definitions to LLM")
    
    # Merge with template - the words_text includes both word and definition for each word
    # The prompt template placeholder <<>> will be replaced with the formatted words and definitions
    full_prompt = merge_prompt_with_words(prompt_template, words_text)
    
    # Query Ollama
    try:
        response = query_ollama(full_prompt)
    except Exception as e:
        print(f"    Error querying Ollama: {e}")
        print(f"    Keeping all words in batch as fallback")
        # On error, keep all words as fallback
        return [w["word"] for w in batch], {}
    
    # Extract JSON from response
    json_str = extract_json_from_response(response)
    if json_str is None:
        print(f"    Warning: Could not extract JSON from response")
        print(f"    Keeping all words in batch as fallback")
        return [w["word"] for w in batch], {}
    
    # Parse JSON
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"    Warning: Invalid JSON in response: {e}")
        print(f"    Response preview: {response[:200]}...")
        print(f"    Keeping all words in batch as fallback")
        return [w["word"] for w in batch], {}
    
    # Extract keep and remove lists - normalize them (lowercase, strip whitespace)
    kept_words_raw = result.get("keep", [])
    removed_words_raw = result.get("remove", [])
    explanations = result.get("explanations", {})
    
    # Normalize LLM responses: lowercase and strip whitespace
    kept_words_normalized = [w.strip().lower() for w in kept_words_raw if w and isinstance(w, str)]
    removed_words_normalized = [w.strip().lower() for w in removed_words_raw if w and isinstance(w, str)]
    
    # Create a case-insensitive mapping from normalized word -> original word object
    # This handles cases where LLM returns "Fear" but original is "fear"
    batch_word_map = {}
    batch_words_set = set()
    for word_obj in batch:
        original_word = word_obj["word"]
        normalized_word = original_word.lower().strip()
        batch_words_set.add(original_word)
        # Map normalized word to original word object
        if normalized_word not in batch_word_map:
            batch_word_map[normalized_word] = word_obj
        # Also map original case (lowercased) for exact matches
        batch_word_map[original_word.lower()] = word_obj
        
        # Handle common variations: if LLM returns a variation (e.g., "ferreting" for "ferret")
        # Try to match base forms - this is a simple heuristic
        # For words ending in -ing, -ed, -er, -s, try matching the base form
        if normalized_word.endswith('ing'):
            base = normalized_word[:-3]
            if base not in batch_word_map:
                batch_word_map[base] = word_obj
        if normalized_word.endswith('ed'):
            base = normalized_word[:-2]
            if base not in batch_word_map:
                batch_word_map[base] = word_obj
        if normalized_word.endswith('er') and len(normalized_word) > 3:
            base = normalized_word[:-2]
            if base not in batch_word_map:
                batch_word_map[base] = word_obj
        if normalized_word.endswith('s') and len(normalized_word) > 2:
            base = normalized_word[:-1]
            if base not in batch_word_map:
                batch_word_map[base] = word_obj
    
    # Match normalized LLM responses back to original words
    matched_kept_words = []
    matched_removed_words = {}
    unmatched_kept = []
    unmatched_removed = []
    
    for normalized_word in kept_words_normalized:
        # Try direct match first
        if normalized_word in batch_word_map:
            original_word = batch_word_map[normalized_word]["word"]
            matched_kept_words.append(original_word)
        else:
            # Try to match if LLM returned a phrase containing the word (e.g., "To fear" -> "fear")
            # Check if any batch word is contained in the normalized response
            matched = False
            for batch_word in batch_words_set:
                batch_word_lower = batch_word.lower()
                # If LLM response contains the batch word, or batch word contains the response
                if normalized_word in batch_word_lower or batch_word_lower in normalized_word:
                    # Prefer exact substring match
                    if normalized_word == batch_word_lower:
                        matched_kept_words.append(batch_word)
                        matched = True
                        break
            if not matched:
                unmatched_kept.append(normalized_word)
    
    for normalized_word in removed_words_normalized:
        # Try direct match first
        if normalized_word in batch_word_map:
            original_word = batch_word_map[normalized_word]["word"]
            # Get explanation (try both normalized and original case)
            explanation = explanations.get(normalized_word) or explanations.get(original_word) or "No explanation provided"
            matched_removed_words[original_word] = explanation
        else:
            # Try to match if LLM returned a phrase containing the word
            matched = False
            for batch_word in batch_words_set:
                batch_word_lower = batch_word.lower()
                if normalized_word == batch_word_lower or (normalized_word in batch_word_lower or batch_word_lower in normalized_word):
                    if normalized_word == batch_word_lower:
                        explanation = explanations.get(normalized_word) or explanations.get(batch_word) or "No explanation provided"
                        matched_removed_words[batch_word] = explanation
                        matched = True
                        break
            if not matched:
                unmatched_removed.append(normalized_word)
                # Still record it with explanation for logging (won't affect final output)
                explanation = explanations.get(normalized_word) or "No explanation provided"
                matched_removed_words[normalized_word] = explanation
    
    # Validate that all words in batch are accounted for
    accounted_for = set(matched_kept_words) | set(matched_removed_words.keys())
    
    # Check for words that were in batch but not in LLM response
    missing_from_response = batch_words_set - accounted_for
    # Check for words in response that don't match batch words (might be variations)
    extra_in_response = set(unmatched_kept) | set(unmatched_removed)
    
    if missing_from_response or extra_in_response:
        print(f"    Warning: Batch words mismatch!")
        if missing_from_response:
            print(f"      Missing from LLM response: {sorted(missing_from_response)}")
            print(f"        (These words will be kept by default)")
            # Keep words that weren't in LLM response
            for word in missing_from_response:
                if word not in matched_kept_words:
                    matched_kept_words.append(word)
        if extra_in_response:
            print(f"      Extra/variation words in response (ignored): {sorted(extra_in_response)}")
    
    print(f"    Result: {len(matched_kept_words)} kept, {len(matched_removed_words)} removed")
    
    return matched_kept_words, matched_removed_words


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Filter and prune words from a spelling bee puzzle JSON using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input JSON file (spelling bee puzzle)"
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output JSON file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="prune_word_list.txt",
        help="Path to the pruning prompt template (default: prune_word_list.txt)"
    )
    parser.add_argument(
        "--frequency-threshold",
        type=float,
        default=MUST_INCLUDE_WORD_FREQUENCY_THRESHOLD,
        help=f"Frequency threshold for always-include words (default: {MUST_INCLUDE_WORD_FREQUENCY_THRESHOLD})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Number of words to process per batch (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=OLLAMA_MODEL,
        help=f"Ollama model to use (default: {OLLAMA_MODEL})"
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=OLLAMA_URL,
        help=f"Ollama API URL (default: {OLLAMA_URL})"
    )
    parser.add_argument(
        "--log-removed",
        type=str,
        default=None,
        help="Path to log file for removed words (default: print to stdout)"
    )
    parser.add_argument(
        "--save-whitelisted",
        type=str,
        default=None,
        help="Path to save whitelisted words separately (before LLM processing)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Word Filtering and Pruning")
    print("=" * 80)
    print()
    
    # Load input JSON
    print(f"Loading puzzle JSON from: {args.input_file}")
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            puzzle_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Extract words list from puzzle data
    if "words" in puzzle_data:
        words_list = puzzle_data["words"]
    elif "candidates" in puzzle_data:
        print("Error: Input file appears to be candidate words JSON, not puzzle JSON.", file=sys.stderr)
        print("This script expects a puzzle JSON file (output from generate_spelling_bee_puzzle.py)", file=sys.stderr)
        print("which contains a 'words' array with word objects having 'word', 'frequency', and 'definition' fields.", file=sys.stderr)
        sys.exit(1)
    elif isinstance(puzzle_data, list):
        # Assume the file is just a list of word objects
        words_list = puzzle_data
    else:
        print("Error: Could not find 'words' array in JSON file.", file=sys.stderr)
        print(f"Found keys: {list(puzzle_data.keys()) if isinstance(puzzle_data, dict) else 'not a dict'}", file=sys.stderr)
        sys.exit(1)
    
    if len(words_list) == 0:
        print("Error: No words found in the input file.", file=sys.stderr)
        sys.exit(1)
    
    print(f"✓ Loaded {len(words_list)} words")
    print()
    
    # First, automatically remove words with frequency 0.0 (invalid/not in dictionary)
    print("Filtering out words with frequency 0.0...")
    words_with_frequency = []
    zero_frequency_words = []
    
    for word_obj in words_list:
        freq = word_obj.get("frequency", 0.0)
        if freq == 0.0:
            zero_frequency_words.append(word_obj["word"])
        else:
            words_with_frequency.append(word_obj)
    
    if zero_frequency_words:
        print(f"✓ Removed {len(zero_frequency_words)} words with frequency 0.0")
        print(f"  Examples: {', '.join(zero_frequency_words[:10])}{'...' if len(zero_frequency_words) > 10 else ''}")
    else:
        print(f"✓ No words with frequency 0.0 found")
    print()
    
    # Classify words by frequency threshold (only for words with frequency > 0.0)
    print(f"Classifying words (frequency threshold: {args.frequency_threshold})...")
    always_include = []
    to_be_processed = []
    
    for word_obj in words_with_frequency:
        freq = word_obj.get("frequency", 0.0)
        if freq > args.frequency_threshold:
            always_include.append(word_obj)
        else:
            to_be_processed.append(word_obj)
    
    print(f"✓ {len(always_include)} words always included (frequency > threshold)")
    print(f"✓ {len(to_be_processed)} words to be processed by LLM")
    print()
    
    # Save whitelisted words separately if requested
    if args.save_whitelisted and always_include:
        print(f"Saving whitelisted words to: {args.save_whitelisted}")
        # If input had puzzle structure, preserve it for whitelisted output
        if "seed_word" in puzzle_data:
            whitelisted_output = puzzle_data.copy()
            whitelisted_output["words"] = always_include
            whitelisted_output["total_words"] = len(always_include)
            whitelisted_output["note"] = "Whitelisted words (frequency > threshold) - saved before LLM processing"
        else:
            whitelisted_output = {
                "words": always_include,
                "total_words": len(always_include),
                "note": "Whitelisted words (frequency > threshold) - saved before LLM processing"
            }
        
        with open(args.save_whitelisted, 'w', encoding='utf-8') as f:
            json.dump(whitelisted_output, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {len(always_include)} whitelisted words")
        print()
    
    # Load prompt template
    print(f"Loading prompt template from: {args.prompt}")
    prompt_template = load_prompt_template(args.prompt)
    print(f"✓ Prompt template loaded ({len(prompt_template)} characters)")
    print()
    
    # Process words in batches
    if not to_be_processed:
        print("No words to process. All words are whitelisted.")
        kept_words_list = []
        all_removed = {}
    else:
        print(f"Processing {len(to_be_processed)} words in batches of {args.batch_size}...")
        print()
        
        all_kept_words = []
        all_removed = {}
        
        total_batches = (len(to_be_processed) + args.batch_size - 1) // args.batch_size
        
        for i in range(0, len(to_be_processed), args.batch_size):
            batch = to_be_processed[i:i + args.batch_size]
            batch_num = (i // args.batch_size) + 1
            
            kept_words, removed_dict = process_batch(
                batch,
                prompt_template,
                batch_num,
                total_batches
            )
            
            all_kept_words.extend(kept_words)
            all_removed.update(removed_dict)
        
        print()
        print(f"✓ Processed {total_batches} batch(es)")
        print(f"  Kept: {len(all_kept_words)} words")
        print(f"  Removed: {len(all_removed)} words")
        print()
        
        # Filter to_be_processed to only include kept words
        # NOTE: LLM returns word strings (now matched/normalized), so we match them 
        # back to original word objects to preserve definitions, frequency, and other metadata
        kept_words_set = set(all_kept_words)
        kept_words_list = [w for w in to_be_processed if w["word"] in kept_words_set]
        
        # Verify that all kept words have their definitions preserved
        if len(kept_words_list) != len(all_kept_words):
            print(f"  Warning: Mismatch in kept words count!")
            print(f"    Expected: {len(all_kept_words)}, Found: {len(kept_words_list)}")
            missing = kept_words_set - {w["word"] for w in kept_words_list}
            if missing:
                print(f"    Missing word objects for: {sorted(missing)}")
                print(f"    These words will not be included in final output")
        else:
            print(f"  ✓ All {len(kept_words_list)} kept words matched and have definitions preserved")
    
    # Combine always_include and kept words
    final_words = always_include + kept_words_list
    
    # Log removed words
    if all_removed:
        log_lines = []
        log_lines.append("=" * 80)
        log_lines.append("REMOVED WORDS")
        log_lines.append("=" * 80)
        log_lines.append(f"Total removed: {len(all_removed)}")
        log_lines.append("")
        
        for word, explanation in sorted(all_removed.items()):
            log_lines.append(f"Word: {word}")
            log_lines.append(f"Reason: {explanation}")
            log_lines.append("")
        
        log_text = "\n".join(log_lines)
        
        if args.log_removed:
            with open(args.log_removed, 'w', encoding='utf-8') as f:
                f.write(log_text)
            print(f"✓ Removed words logged to: {args.log_removed}")
        else:
            print(log_text)
        print()
    
    # Prepare output structure
    # If input had puzzle structure, preserve it; otherwise create simple structure
    if "seed_word" in puzzle_data:
        output_data = puzzle_data.copy()
        output_data["words"] = final_words
        output_data["total_words"] = len(final_words)
        output_data["original_word_count"] = len(words_list)
        output_data["filtering_stats"] = {
            "removed_zero_frequency": len(zero_frequency_words),
            "always_included": len(always_include),
            "processed_by_llm": len(to_be_processed),
            "kept_by_llm": len(kept_words_list),
            "removed_by_llm": len(all_removed)
        }
    else:
        output_data = {
            "words": final_words,
            "total_words": len(final_words),
            "original_word_count": len(words_list),
            "filtering_stats": {
                "removed_zero_frequency": len(zero_frequency_words),
                "always_included": len(always_include),
                "processed_by_llm": len(to_be_processed),
                "kept_by_llm": len(kept_words_list),
                "removed_by_llm": len(all_removed)
            }
        }
    
    # Save output
    print(f"Saving filtered words to: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(final_words)} words to output file")
    print()
    
    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Original words: {len(words_list)}")
    print(f"  - Removed (frequency = 0.0): {len(zero_frequency_words)}")
    print(f"  - Always included (frequency > {args.frequency_threshold}): {len(always_include)}")
    print(f"  - Processed by LLM: {len(to_be_processed)}")
    print(f"    * Kept: {len(kept_words_list)}")
    print(f"    * Removed: {len(all_removed)}")
    print(f"Final words: {len(final_words)}")
    if len(words_list) > 0:
        removed_count = len(words_list) - len(final_words)
        removed_percent = 100 * removed_count / len(words_list)
        print(f"Total removed: {removed_count} ({removed_percent:.1f}%)")
        print(f"  - Zero frequency: {len(zero_frequency_words)}")
        print(f"  - Removed by LLM: {len(all_removed)}")
    else:
        print("Total removed: 0 (N/A - no words in input)")
    print("=" * 80)


if __name__ == "__main__":
    main()
