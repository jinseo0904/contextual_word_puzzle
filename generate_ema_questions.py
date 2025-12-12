#!/usr/bin/env python3
"""
Script to generate EMA questions and clues for words using Ollama.
Takes a word list and generates smartwatch-friendly EMA questions with clues.
"""

import json
import requests
import argparse
import sys
import re
from typing import Dict, List, Any, Optional
from pathlib import Path


def load_prompt_template(prompt_path: str) -> str:
    """Load the prompt template from file."""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Prompt template file not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading prompt template: {e}", file=sys.stderr)
        sys.exit(1)


def load_word_list(word_list_path: str) -> List[str]:
    """Load the word list from a plain text file (one word per line)."""
    try:
        with open(word_list_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]
        return words
    except FileNotFoundError:
        print(f"Error: Word list file not found: {word_list_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading word list: {e}", file=sys.stderr)
        sys.exit(1)


def load_pruned_words_json(json_path: str) -> List[str]:
    """Load pruned words from JSON file and return list of word strings."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Extract words from the JSON structure
    if "words" in data:
        words_list = data["words"]
        if isinstance(words_list, list) and len(words_list) > 0:
            if isinstance(words_list[0], dict) and "word" in words_list[0]:
                # List of word objects: extract just the word strings
                return [word_obj["word"] for word_obj in words_list]
            elif isinstance(words_list[0], str):
                # Already a list of strings
                return words_list
    
    raise ValueError(f"Could not extract words from JSON file. Expected structure with 'words' array containing word objects.")


def build_full_prompt(template: str, word_list: List[str]) -> str:
    """Build the complete prompt by replacing the word list placeholder."""
    word_list_text = '\n'.join(word_list)
    
    # Replace placeholder
    if '<<<WORD_LIST>>>' in template:
        return template.replace('<<<WORD_LIST>>>', word_list_text)
    elif '<<>>' in template:
        # Handle <<>> placeholder format
        return template.replace('<<>>', word_list_text)
    else:
        # If no placeholder found, append the word list
        return template + "\n\n" + word_list_text


def query_ollama(prompt: str, model: str = "gpt-oss:20b", ollama_url: str = "http://localhost:11434") -> str:
    """Send prompt to Ollama and get response."""
    url = f"{ollama_url}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    print(f"Sending request to Ollama ({model})...")
    print("This may take a while...\n")
    
    try:
        response = requests.post(url, json=payload, timeout=600)  # 10 minute timeout for large models
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', '')
    
    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"Could not connect to Ollama at {ollama_url}. Make sure Ollama is running.")
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Request to Ollama timed out after 600 seconds.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error querying Ollama: {e}")


def parse_json_response(response: str) -> Optional[List[Dict[str, Any]]]:
    """Extract and parse JSON array from the response.
    
    The expected output is a JSON array of word objects.
    """
    if not response or not response.strip():
        print("Warning: Empty response from LLM")
        return None
    
    # Try to find JSON in markdown code blocks first
    json_match = re.search(r'```(?:json)?\s*(\[.*?\])', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON array directly - look for balanced brackets
        start_idx = response.find('[')
        if start_idx == -1:
            print("Warning: No opening bracket found in response")
            return None
        
        # Find the matching closing bracket by counting brackets
        bracket_count = 0
        end_idx = start_idx
        for i in range(start_idx, len(response)):
            if response[i] == '[':
                bracket_count += 1
            elif response[i] == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    end_idx = i + 1
                    break
        
        if bracket_count != 0:
            print("Warning: Unbalanced brackets in JSON response")
            # Fallback to last closing bracket
            end_idx = response.rfind(']') + 1
            if end_idx == 0:
                print("Warning: No closing bracket found")
                return None
        
        json_str = response[start_idx:end_idx]
    
    try:
        parsed = json.loads(json_str)
        # Validate structure
        if not isinstance(parsed, list):
            print("Warning: JSON root is not an array")
            return None
        
        # Validate each entry
        validated = []
        for i, entry in enumerate(parsed):
            if not isinstance(entry, dict):
                print(f"Warning: Entry {i} is not an object, skipping")
                continue
            
            if 'word' not in entry or 'usable' not in entry:
                print(f"Warning: Entry {i} missing required fields (word, usable), skipping")
                continue
            
            validated.append(entry)
        
        if len(validated) != len(parsed):
            print(f"Warning: Only {len(validated)}/{len(parsed)} entries were valid")
        
        return validated
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse JSON: {e}")
        print(f"JSON string preview (first 500 chars): {json_str[:500]}...")
        if len(json_str) > 500:
            print(f"... (truncated, total length: {len(json_str)} chars)")
        return None


def validate_output(output: List[Dict[str, Any]], input_words: List[str]) -> Dict[str, Any]:
    """Validate that the output contains all input words and is well-formed."""
    validation_results = {
        'missing_words': [],
        'extra_words': [],
        'invalid_entries': [],
        'usability_stats': {'usable': 0, 'unusable': 0}
    }
    
    output_words = set()
    for entry in output:
        word = entry.get('word', '')
        output_words.add(word)
        
        usable = entry.get('usable', None)
        if usable is True:
            validation_results['usability_stats']['usable'] += 1
            # Check required fields for usable words
            required_fields = ['construct', 'ema_question', 'options', 'generic_clue', 'response_clues']
            missing = [f for f in required_fields if f not in entry]
            if missing:
                validation_results['invalid_entries'].append({
                    'word': word,
                    'issue': f"Missing required fields: {missing}"
                })
        elif usable is False:
            validation_results['usability_stats']['unusable'] += 1
            # Check required field for unusable words
            if 'reason' not in entry:
                validation_results['invalid_entries'].append({
                    'word': word,
                    'issue': "Missing 'reason' field for unusable word"
                })
        else:
            validation_results['invalid_entries'].append({
                'word': word,
                'issue': f"Invalid 'usable' value: {usable}"
            })
    
    input_words_set = set(input_words)
    validation_results['missing_words'] = list(input_words_set - output_words)
    validation_results['extra_words'] = list(output_words - input_words_set)
    
    return validation_results


def display_results(output: List[Dict[str, Any]], validation: Dict[str, Any]):
    """Display the results in a formatted way."""
    print("=" * 80)
    print("GENERATED EMA QUESTIONS:")
    print("=" * 80)
    
    usable_words = [w for w in output if w.get('usable') is True]
    unusable_words = [w for w in output if w.get('usable') is False]
    
    print(f"\nTotal words processed: {len(output)}")
    print(f"  - Usable: {len(usable_words)}")
    print(f"  - Unusable: {len(unusable_words)}")
    print()
    
    if validation['missing_words']:
        print(f"⚠ Missing words from output: {validation['missing_words']}")
        print()
    
    if validation['extra_words']:
        print(f"⚠ Extra words in output: {validation['extra_words']}")
        print()
    
    if validation['invalid_entries']:
        print(f"⚠ Invalid entries:")
        for issue in validation['invalid_entries']:
            print(f"  - {issue['word']}: {issue['issue']}")
        print()
    
    # Display examples
    if usable_words:
        print("Examples of usable words:")
        for i, entry in enumerate(usable_words[:3], 1):
            print(f"\n{i}. Word: {entry['word'].upper()}")
            print(f"   Construct: {entry.get('construct', 'N/A')}")
            print(f"   EMA Question: {entry.get('ema_question', 'N/A')}")
            print(f"   Options: {entry.get('options', [])}")
            print(f"   Generic Clue: {entry.get('generic_clue', 'N/A')}")
            if 'response_clues' in entry:
                print(f"   Response Clues:")
                for option, clue in entry['response_clues'].items():
                    print(f"     - If '{option}': {clue}")
    
    if unusable_words:
        print(f"\nExamples of unusable words:")
        for i, entry in enumerate(unusable_words[:3], 1):
            print(f"\n{i}. Word: {entry['word'].upper()}")
            print(f"   Reason: {entry.get('reason', 'N/A')}")
    
    print()
    print("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Generate EMA questions and clues for words using Ollama',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_ema_questions.py --words word_list.txt --output ema_questions.json
  python generate_ema_questions.py --pruned-words pruned.json --output ema_questions.json
  python generate_ema_questions.py --words words.txt --model gemma3:27b --dry-run
        """
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='generate_priming_ema_questions.txt',
        help='Path to prompt template file (default: generate_priming_ema_questions.txt)'
    )
    parser.add_argument(
        '--words',
        type=str,
        default=None,
        help='Path to word list file (plain text, one word per line)'
    )
    parser.add_argument(
        '--pruned-words', '--pruned_words',
        dest='pruned_words',
        type=str,
        default=None,
        help='Path to pruned words JSON file (output from filter_and_prune_words.py)'
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
        default='generated_ema_questions.json',
        help='Output file for generated EMA questions (JSON) (default: generated_ema_questions.json)'
    )
    parser.add_argument(
        '--save-prompt',
        type=str,
        default=None,
        help='File to save the full prompt for inspection'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only generate and save the prompt, do not call Ollama'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("EMA Questions Generation")
    print("=" * 80)
    print()
    
    # Load prompt template
    print(f"Loading prompt template from: {args.prompt}")
    template = load_prompt_template(args.prompt)
    print(f"✓ Prompt template loaded ({len(template)} characters)")
    print()
    
    # Load words - either from pruned JSON or plain text file
    if args.pruned_words:
        print(f"Loading pruned words from JSON: {args.pruned_words}")
        words = load_pruned_words_json(args.pruned_words)
        print(f"✓ Loaded {len(words)} words from pruned JSON")
    elif args.words:
        print(f"Loading word list from: {args.words}")
        words = load_word_list(args.words)
        print(f"✓ Loaded {len(words)} words")
    else:
        print("Error: Must provide either --words or --pruned-words", file=sys.stderr)
        sys.exit(1)
    print()
    
    # Build prompt
    print("Building full prompt...")
    full_prompt = build_full_prompt(template, words)
    print(f"✓ Full prompt created ({len(full_prompt)} characters)")
    print()
    
    # Save the prompt for inspection
    if args.save_prompt:
        with open(args.save_prompt, 'w', encoding='utf-8') as f:
            f.write(full_prompt)
        print(f"✓ Saved full prompt to '{args.save_prompt}'")
        print()
    elif not args.dry_run:
        # Auto-save prompt for debugging
        prompt_file = args.output.replace('.json', '_prompt.txt')
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(full_prompt)
        print(f"✓ Saved full prompt to '{prompt_file}' for inspection")
        print()
    
    # Show preview
    print("=" * 80)
    print("PROMPT PREVIEW:")
    print("=" * 80)
    print(f"Word list length: {len(words)} words")
    print(f"\nFirst 10 words: {', '.join(words[:10])}")
    if len(words) > 10:
        print(f"... ({len(words) - 10} more)")
    print()
    
    if args.dry_run:
        print("=" * 80)
        print("DRY RUN MODE - Skipping Ollama API call")
        print("=" * 80)
        if args.save_prompt:
            print(f"✓ Full prompt saved to '{args.save_prompt}'")
        else:
            print(f"✓ Full prompt saved to '{prompt_file}'")
        print("  You can review it before running with the API.")
        return
    
    # Query Ollama
    print("=" * 80)
    print("QUERYING OLLAMA")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"URL: {args.ollama_url}")
    print()
    
    try:
        response = query_ollama(full_prompt, model=args.model, ollama_url=args.ollama_url)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Parse JSON response
    print("\nParsing JSON response...")
    output = parse_json_response(response)
    
    if output is None:
        print("\nError: Could not parse JSON from response")
        print("\nRaw response:")
        print(response)
        # Save raw response for debugging
        raw_file = args.output.replace('.json', '_raw.txt')
        with open(raw_file, 'w', encoding='utf-8') as f:
            f.write(response)
        print(f"\nRaw response saved to: {raw_file}")
        sys.exit(1)
    
    # Validate output
    print("Validating output...")
    validation = validate_output(output, words)
    print(f"✓ Parsed {len(output)} word entries")
    print()
    
    # Display results
    display_results(output, validation)
    
    # Save output
    print(f"Saving results to: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(output)} entries to '{args.output}'")
    print()
    
    print("=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
