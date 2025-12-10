#!/usr/bin/env python3
"""
Test script for word clue generation using Ollama's gemma3:27b model.
Combines the word clue prompt with a day narrative and word list.
Can work with pruned words JSON from filter_and_prune_words.py pipeline.
"""

import json
import requests
import argparse
import sys
from typing import Dict, List, Any


def load_prompt_template(prompt_path: str) -> str:
    """Load the prompt template from file."""
    with open(prompt_path, 'r') as f:
        return f.read()


def load_narrative(narrative_path: str) -> str:
    """Load the day narrative from file."""
    with open(narrative_path, 'r') as f:
        return f.read().strip()


def load_word_list(word_list_path: str) -> str:
    """Load the word list from file and return as formatted string."""
    with open(word_list_path, 'r') as f:
        words = [line.strip() for line in f if line.strip()]
    return '\n'.join(words)


def load_pruned_words_json(json_path: str) -> List[str]:
    """Load pruned words from JSON file and return list of word strings."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract words from the JSON structure
    if "words" in data:
        # It's a puzzle JSON with words array
        words_list = data["words"]
        if isinstance(words_list, list) and len(words_list) > 0:
            if isinstance(words_list[0], dict) and "word" in words_list[0]:
                # List of word objects: extract just the word strings
                return [word_obj["word"] for word_obj in words_list]
            elif isinstance(words_list[0], str):
                # Already a list of strings
                return words_list
    
    # Fallback: try to extract from other possible structures
    raise ValueError(f"Could not extract words from JSON file. Expected structure with 'words' array containing word objects.")


def build_full_prompt(template: str, narrative: str, word_list: str) -> str:
    """Build the complete prompt by replacing placeholders."""
    full_prompt = template.replace('<<<DAY_SUMMARY>>>', narrative)
    full_prompt = full_prompt.replace('<<<WORD_LIST>>>', word_list)
    return full_prompt


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


def parse_json_response(response: str) -> Dict[str, Any]:
    """Try to extract and parse JSON from the response."""
    import re
    
    # Try to find JSON in markdown code blocks first
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON object directly
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            return None
        json_str = response[start_idx:end_idx]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse JSON: {e}")
        print(f"JSON string preview: {json_str[:200]}...")
        return None


def display_results(response: str, parsed_json: Dict[str, Any] = None):
    """Display the results in a formatted way."""
    print("=" * 80)
    print("GENERATED WORD CLUES:")
    print("=" * 80)
    
    if parsed_json and 'selected_words' in parsed_json:
        selected_words = parsed_json['selected_words']
        print(f"\nSelected {len(selected_words)} words from the list:\n")
        
        for i, word_data in enumerate(selected_words, 1):
            word = word_data.get('word', 'N/A')
            clue = word_data.get('clue', 'N/A')
            reasoning = word_data.get('reasoning', 'N/A')
            
            print(f"{i}. Word: {word.upper()}")
            print(f"   Clue: {clue}")
            print(f"   Reasoning: {reasoning}")
            print()
    else:
        print("\nRaw response (JSON parsing failed):")
        print(response)
    
    print("=" * 80)


def main():
    """Main function to test the word clue prompt."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Test word clue generation with Ollama'
    )
    parser.add_argument(
        '--model', '-m',
        default='gemma3:27b',
        help='Ollama model to use (default: gemma3:27b)'
    )
    parser.add_argument(
        '--ollama-url',
        default='http://localhost:11434',
        help='Ollama API URL (default: http://localhost:11434)'
    )
    parser.add_argument(
        '--prompt',
        default='generate_contextual_word_clues.txt',
        help='Path to prompt template file'
    )
    parser.add_argument(
        '--narrative',
        default='generated_narrative.txt',
        help='Path to day narrative file'
    )
    parser.add_argument(
        '--words',
        default=None,
        help='Path to word list file (plain text, one word per line)'
    )
    parser.add_argument(
        '--pruned-words', '--pruned_words',
        dest='pruned_words',
        default=None,
        help='Path to pruned words JSON file (output from filter_and_prune_words.py)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only generate and save the prompt, do not call Ollama'
    )
    parser.add_argument(
        '--output',
        default='generated_word_clues.json',
        help='Output file for generated clues (JSON)'
    )
    parser.add_argument(
        '--save-prompt',
        default='generated_word_prompt.txt',
        help='File to save the full prompt for inspection'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Word Clue Generation Test")
    print("=" * 80)
    print()
    
    # Load inputs
    print(f"Loading prompt template from: {args.prompt}")
    template = load_prompt_template(args.prompt)
    
    print(f"Loading narrative from: {args.narrative}")
    narrative = load_narrative(args.narrative)
    
    # Load words - either from pruned JSON or plain text file
    if args.pruned_words:
        print(f"Loading pruned words from JSON: {args.pruned_words}")
        words = load_pruned_words_json(args.pruned_words)
        word_list = '\n'.join(words)
        word_count = len(words)
        print(f"✓ Loaded {word_count} words from pruned JSON")
    elif args.words:
        print(f"Loading word list from: {args.words}")
        word_list = load_word_list(args.words)
        word_count = len([w for w in word_list.split('\n') if w.strip()])
        print(f"✓ Loaded {word_count} candidate words")
    else:
        print("Error: Must provide either --words or --pruned-words", file=sys.stderr)
        sys.exit(1)
    print()
    
    # Build prompt
    print("Building full prompt...")
    full_prompt = build_full_prompt(template, narrative, word_list)
    
    # Save the prompt for inspection
    with open(args.save_prompt, 'w') as f:
        f.write(full_prompt)
    print(f"✓ Saved full prompt to '{args.save_prompt}' for inspection")
    print()
    
    # Show preview
    print("=" * 80)
    print("PROMPT PREVIEW:")
    print("=" * 80)
    print(f"Day narrative length: {len(narrative)} characters")
    print(f"Word list length: {word_count} words")
    print()
    print("First 500 chars of narrative:")
    print(narrative[:500])
    print("...")
    print()
    
    if args.dry_run:
        print("=" * 80)
        print("DRY RUN MODE - Skipping Ollama API call")
        print("=" * 80)
        print(f"✓ Full prompt saved to '{args.save_prompt}'")
        print("  You can review it before running with the API.")
        return
    
    # Query Ollama
    print("=" * 80)
    print("QUERYING OLLAMA")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"URL: {args.ollama_url}")
    try:
        response = query_ollama(full_prompt, model=args.model, ollama_url=args.ollama_url)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Try to parse JSON
    parsed_json = parse_json_response(response)
    
    # Display results
    print()
    display_results(response, parsed_json)
    
    # Save response
    if parsed_json:
        with open(args.output, 'w') as f:
            json.dump(parsed_json, f, indent=2)
        print(f"✓ Saved clues to '{args.output}'")
    else:
        # Save raw response if JSON parsing failed
        with open(args.output.replace('.json', '_raw.txt'), 'w') as f:
            f.write(response)
        print(f"✓ Saved raw response to '{args.output.replace('.json', '_raw.txt')}'")
    
    print()
    print("=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
