#!/usr/bin/env python3
"""
Script to merge a diary narrative with a prompt template and run it on Ollama.
Takes a path to a diary narrative txt file, merges it with the template,
and runs the finalized prompt on the gpt-oss:20b model.
"""

import argparse
import json
import re
import requests
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional


def load_template(template_path: str) -> str:
    """Load the prompt template from file."""
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Template file not found: {template_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading template file: {e}", file=sys.stderr)
        sys.exit(1)


def load_diary_narrative(narrative_path: str) -> str:
    """Load the diary narrative from file."""
    try:
        with open(narrative_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: Diary narrative file not found: {narrative_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading diary narrative file: {e}", file=sys.stderr)
        sys.exit(1)


def merge_prompt(template: str, narrative: str) -> str:
    """Merge the template with the diary narrative.
    
    If the template contains placeholders like <<<DIARY_NARRATIVE>>> or <<<NARRATIVE>>>,
    they will be replaced with the narrative text.
    Otherwise, the narrative will be appended to the template.
    """
    # Common placeholder patterns
    placeholders = ['<<<DIARY_NARRATIVE>>>', '<<<NARRATIVE>>>', '<<<DAY_SUMMARY>>>', '<<>>']
    
    merged = template
    replaced = False
    
    for placeholder in placeholders:
        if placeholder in merged:
            merged = merged.replace(placeholder, narrative)
            replaced = True
    
    # If no placeholder was found, append the narrative
    if not replaced:
        if template.strip():
            merged = template + "\n\n" + narrative
        else:
            merged = narrative
    
    return merged


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


def validate_candidate(candidate: Dict[str, Any], index: int, narrative: str) -> List[str]:
    """Validate a single candidate word and return list of errors."""
    errors = []
    
    # Check required fields
    if 'word' not in candidate:
        errors.append(f"Candidate {index}: Missing 'word' field")
        return errors
    
    if 'distinct_letters' not in candidate:
        errors.append(f"Candidate {index}: Missing 'distinct_letters' field")
        return errors
    
    if 'clue' not in candidate:
        errors.append(f"Candidate {index}: Missing 'clue' field")
        return errors
    
    word = candidate['word'].strip().lower()
    distinct_letters_raw = candidate['distinct_letters']
    clue = candidate['clue']
    
    # Validate word is not empty
    if not word:
        errors.append(f"Candidate {index}: 'word' is empty")
    
    # Validate word length (must be at least 7 characters)
    if len(word) < 7:
        errors.append(f"Candidate {index}: Word '{word}' is too short (must be at least 7 characters)")
    
    # Handle distinct_letters - can be a list or string
    if isinstance(distinct_letters_raw, list):
        distinct_letters = [str(letter).lower().strip() for letter in distinct_letters_raw]
    elif isinstance(distinct_letters_raw, str):
        # Handle string like "['w', 'a', 'l', 'k', 'i', 'n', 'g']" or "w, a, l, k, i, n, g"
        distinct_letters = [letter.lower().strip().strip("'\"") 
                           for letter in re.findall(r"[a-zA-Z]", distinct_letters_raw)]
    else:
        errors.append(f"Candidate {index}: 'distinct_letters' must be a list or string")
        return errors
    
    # Validate exactly 7 distinct letters
    if len(distinct_letters) != 7:
        errors.append(f"Candidate {index}: Word '{word}' has {len(distinct_letters)} distinct letters, expected 7")
    
    # Check for duplicates in distinct_letters
    if len(set(distinct_letters)) != len(distinct_letters):
        errors.append(f"Candidate {index}: Word '{word}' has duplicate letters in distinct_letters list")
        distinct_letters = list(set(distinct_letters))  # Deduplicate for further checks
    
    # Validate that distinct_letters actually match the letters in the word
    word_letters = set(word)
    distinct_letters_set = set(distinct_letters)
    
    # Check that all distinct_letters are in the word
    missing_letters = distinct_letters_set - word_letters
    if missing_letters:
        errors.append(f"Candidate {index}: Word '{word}' does not contain distinct letters: {sorted(missing_letters)}")
    
    # Check that the word has exactly 7 distinct letters (not more, not less)
    if len(word_letters) != 7:
        errors.append(f"Candidate {index}: Word '{word}' actually has {len(word_letters)} distinct letters, not 7")
    
    # Check that distinct_letters exactly matches the word's distinct letters
    if word_letters != distinct_letters_set:
        extra_in_word = word_letters - distinct_letters_set
        if extra_in_word:
            errors.append(f"Candidate {index}: Word '{word}' contains extra distinct letters not in the list: {sorted(extra_in_word)}")
        if missing_letters:
            errors.append(f"Candidate {index}: distinct_letters list contains letters not in the word: {sorted(missing_letters)}")
    
    # Validate clue is not empty
    if not clue or not clue.strip():
        errors.append(f"Candidate {index}: 'clue' is empty")
    
    return errors


def sanity_check_response(response: str, narrative: str) -> Tuple[bool, Optional[Dict[str, Any]], List[str]]:
    """Perform sanity checks on the model's JSON response.
    
    Returns:
        Tuple of (is_valid, parsed_json, list_of_errors)
    """
    errors = []
    
    # Try to extract JSON from response
    json_str = extract_json_from_response(response)
    if json_str is None:
        return False, None, ["Could not find JSON in response. The response may not be valid JSON."]
    
    # Parse JSON
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        return False, None, [f"Invalid JSON: {e}"]
    
    # Check top-level structure
    if not isinstance(parsed, dict):
        return False, None, ["Response is not a JSON object"]
    
    if 'candidates' not in parsed:
        return False, None, ["Missing 'candidates' field in response"]
    
    if not isinstance(parsed['candidates'], list):
        return False, None, ["'candidates' must be a list"]
    
    candidates = parsed['candidates']
    
    # Check candidate count
    if len(candidates) < 3:
        errors.append(f"Only {len(candidates)} candidate(s) provided, expected 3-4")
    elif len(candidates) > 4:
        errors.append(f"{len(candidates)} candidates provided, expected 3-4")
    
    # Validate each candidate
    for i, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            errors.append(f"Candidate {i+1}: Not a valid object")
            continue
        
        candidate_errors = validate_candidate(candidate, i+1, narrative)
        errors.extend(candidate_errors)
    
    is_valid = len(errors) == 0
    return is_valid, parsed, errors


def query_ollama(prompt: str, model: str = "gpt-oss:20b", ollama_url: str = "http://localhost:11434") -> str:
    """Send prompt to Ollama and get response."""
    url = f"{ollama_url}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    print(f"Sending request to Ollama (model: {model})...")
    print("This may take a while...\n")
    
    try:
        response = requests.post(url, json=payload, timeout=600)  # 10 minute timeout for large models
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', '')
    
    except requests.exceptions.ConnectionError:
        error_msg = f"Error: Could not connect to Ollama at {ollama_url}.\n"
        error_msg += "Make sure Ollama is running and accessible."
        return error_msg
    except requests.exceptions.Timeout:
        return "Error: Request to Ollama timed out. The model may be processing a large prompt."
    except requests.exceptions.RequestException as e:
        return f"Error querying Ollama: {e}"


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Merge diary narrative with template and run on Ollama',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_contextual_seed_word.py narrative.txt
  python create_contextual_seed_word.py narrative.txt --template custom_template.txt
  python create_contextual_seed_word.py narrative.txt --output result.txt --dry-run
        """
    )
    parser.add_argument(
        'narrative',
        type=str,
        help='Path to diary narrative txt file'
    )
    parser.add_argument(
        '--template', '-t',
        type=str,
        default='create_contextual_seed_word_template.txt',
        help='Path to prompt template file (default: create_contextual_seed_word_template.txt)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='gpt-oss:20b',
        help='Ollama model to use (default: gpt-oss:20b)'
    )
    parser.add_argument(
        '--ollama-url',
        type=str,
        default='http://localhost:11434',
        help='Ollama API URL (default: http://localhost:11434)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file to save the response (default: print to stdout)'
    )
    parser.add_argument(
        '--save-prompt',
        type=str,
        default=None,
        help='Save the merged prompt to a file before running'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only generate and save the merged prompt, do not call Ollama'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip sanity check validation of the response'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Contextual Seed Word Generation")
    print("=" * 80)
    print()
    
    # Load template
    print(f"Loading template from: {args.template}")
    template = load_template(args.template)
    print(f"✓ Template loaded ({len(template)} characters)")
    
    # Load diary narrative
    print(f"Loading diary narrative from: {args.narrative}")
    narrative = load_diary_narrative(args.narrative)
    print(f"✓ Narrative loaded ({len(narrative)} characters)")
    print()
    
    # Merge prompt
    print("Merging template with narrative...")
    merged_prompt = merge_prompt(template, narrative)
    print(f"✓ Merged prompt created ({len(merged_prompt)} characters)")
    print()
    
    # Save merged prompt if requested
    if args.save_prompt:
        with open(args.save_prompt, 'w', encoding='utf-8') as f:
            f.write(merged_prompt)
        print(f"✓ Saved merged prompt to: {args.save_prompt}")
        print()
    
    # Show preview of the merged prompt
    print("=" * 80)
    print("MERGED PROMPT PREVIEW (first 500 chars):")
    print("=" * 80)
    print(merged_prompt[:500])
    if len(merged_prompt) > 500:
        print("...\n")
    else:
        print()
    
    if args.dry_run:
        print("=" * 80)
        print("DRY RUN MODE - Skipping Ollama API call")
        print("=" * 80)
        if args.save_prompt:
            print(f"✓ Merged prompt saved to: {args.save_prompt}")
        else:
            print("  Use --save-prompt to save the merged prompt to a file.")
        return
    
    # Query Ollama
    print("=" * 80)
    print("QUERYING OLLAMA")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"URL: {args.ollama_url}")
    print()
    
    response = query_ollama(merged_prompt, model=args.model, ollama_url=args.ollama_url)
    
    # Display raw response
    print("\n" + "=" * 80)
    print("RAW RESPONSE:")
    print("=" * 80)
    print(response)
    print()
    
    # Perform sanity check (unless skipped)
    is_valid = None
    parsed_json = None
    errors = []
    
    if not args.skip_validation:
        print("=" * 80)
        print("SANITY CHECK:")
        print("=" * 80)
        is_valid, parsed_json, errors = sanity_check_response(response, narrative)
        
        if is_valid:
            print("✓ All sanity checks passed!")
            print()
            print("Validated JSON structure:")
            print(json.dumps(parsed_json, indent=2))
        else:
            print("✗ Sanity check failed with the following errors:")
            print()
            for error in errors:
                print(f"  • {error}")
            print()
            print("Raw response saved, but validation failed.")
            print("The model may have hallucinated or made errors in the response.")
        
        print()
    else:
        print("=" * 80)
        print("SANITY CHECK: SKIPPED")
        print("=" * 80)
        print("(Use without --skip-validation to enable validation)")
        print()
    
    # Save response if requested
    import datetime
    import os

    # Define the output directory for JSONs
    output_dir = "/home/mhealth-admin/jin/words_with_friends/spelling_bee/generated_jsons"
    os.makedirs(output_dir, exist_ok=True)

    # Get current timestamp in YYYY-MM-DD-HH-MM format
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

    if args.output:
        # Compose filenames with timestamp
        raw_output_filename = os.path.splitext(os.path.basename(args.output))[0]
        raw_json_path = os.path.join(output_dir, f"{raw_output_filename}_raw_{timestamp}.json")
        # with open(raw_json_path, 'w', encoding='utf-8') as f:
        #     f.write(response)
        # print(f"✓ Saved raw response to: {raw_json_path}")
        
        # Also save validated JSON if valid
        if is_valid and parsed_json:
            validated_json_path = os.path.join(output_dir, f"{raw_output_filename}_validated_{timestamp}.json")
            with open(validated_json_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_json, f, indent=2)
            print(f"✓ Saved validated JSON to: {validated_json_path}")
        print()
    
    print("=" * 80)
    if args.skip_validation:
        print("Done!")
    elif is_valid:
        print("Done! Response validated successfully.")
    else:
        print("Done! (with validation errors - review the output above)")
    print("=" * 80)


if __name__ == "__main__":
    main()
