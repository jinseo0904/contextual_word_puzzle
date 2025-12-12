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
import time
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
    """Build the complete prompt by replacing placeholders.
    
    Handles both old format (<<<DAY_SUMMARY>>>, <<<WORD_LIST>>>) 
    and new format (<<>> appears twice - first for day summary, second for word list).
    """
    # Check for new format with <<>> placeholders
    if '<<>>' in template:
        # Count occurrences to determine replacement order
        occurrences = template.count('<<>>')
        if occurrences >= 2:
            # Replace first <<>> with narrative, second with word_list
            parts = template.split('<<>>', 2)
            if len(parts) == 3:
                full_prompt = parts[0] + narrative + parts[1] + word_list + parts[2]
            else:
                # Fallback: replace all occurrences in order
                full_prompt = template.replace('<<>>', narrative, 1)
                full_prompt = full_prompt.replace('<<>>', word_list, 1)
                # If there are more, they'll remain (shouldn't happen)
        elif occurrences == 1:
            # Only one placeholder - assume it's for the word list (day summary might be elsewhere)
            full_prompt = template.replace('<<>>', word_list)
            # Check if narrative is already in template or needs to be added
            if narrative not in full_prompt:
                print("Warning: Only one <<>> placeholder found, assuming it's for word list")
        else:
            full_prompt = template
    else:
        # Old format with explicit placeholders
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
    """Try to extract and parse JSON from the response.
    
    Handles various response formats:
    - JSON in markdown code blocks (```json ... ``` or ``` ... ```)
    - Plain JSON object or array
    - JSON with extra text before/after
    - Arrays will be wrapped in {'selected_words': [...]}
    """
    import re
    
    if not response or not response.strip():
        print("Warning: Empty response from LLM")
        return None
    
    # Try to find JSON in markdown code blocks first
    # Look for ```json or ``` followed by content and closing ```
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)```'
    code_block_match = re.search(code_block_pattern, response, re.DOTALL)
    
    if code_block_match:
        # Extract content from code block
        code_content = code_block_match.group(1).strip()
        # Try to parse the entire code block content as JSON
        json_str = code_content
    else:
        # No code block found, try to find JSON directly
        # Check for both objects { and arrays [
        obj_start = response.find('{')
        arr_start = response.find('[')
        
        # Determine which comes first and what type we're dealing with
        if obj_start == -1 and arr_start == -1:
            print("Warning: No JSON object or array found in response")
            return None
        
        if arr_start != -1 and (obj_start == -1 or arr_start < obj_start):
            # Array comes first or is the only JSON structure
            start_idx = arr_start
            is_array = True
            open_char = '['
            close_char = ']'
        else:
            # Object comes first
            start_idx = obj_start
            is_array = False
            open_char = '{'
            close_char = '}'
        
        # Find the matching closing bracket/brace by counting
        bracket_count = 0
        end_idx = start_idx
        for i in range(start_idx, len(response)):
            if response[i] == open_char:
                bracket_count += 1
            elif response[i] == close_char:
                bracket_count -= 1
                if bracket_count == 0:
                    end_idx = i + 1
                    break
        
        if bracket_count != 0:
            print(f"Warning: Unbalanced {open_char}{close_char} in JSON response")
            # Fallback to last closing bracket/brace
            end_idx = response.rfind(close_char) + 1
            if end_idx == 0:
                print(f"Warning: No closing {close_char} found")
                return None
        
        json_str = response[start_idx:end_idx]
    
    try:
        parsed = json.loads(json_str)
        
        # Handle arrays - wrap them in the expected format
        if isinstance(parsed, list):
            print("Note: Response is an array, wrapping in {'selected_words': [...]}")
            parsed = {'selected_words': parsed}
        
        # Validate structure
        if not isinstance(parsed, dict):
            print(f"Warning: JSON root is not an object or array (type: {type(parsed)})")
            return None
        
        if 'selected_words' not in parsed:
            print("Warning: JSON does not contain 'selected_words' key")
            print(f"Found keys: {list(parsed.keys())}")
            return None
        
        return parsed
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse JSON: {e}")
        print(f"JSON string preview (first 500 chars): {json_str[:500]}...")
        if len(json_str) > 500:
            print(f"... (truncated, total length: {len(json_str)} chars)")
        return None


def display_results(response: str, parsed_json: Dict[str, Any] = None):
    """Display the results in a formatted way."""
    print("=" * 80)
    print("GENERATED WORD CLUES:")
    print("=" * 80)
    
    if parsed_json and 'selected_words' in parsed_json:
        selected_words = parsed_json['selected_words']
        
        if not isinstance(selected_words, list):
            print(f"\nWarning: 'selected_words' is not a list (type: {type(selected_words)})")
            print("\nRaw response:")
            print(response)
            print("=" * 80)
            return
        
        print(f"\nSelected {len(selected_words)} words from the list:\n")
        
        for i, word_data in enumerate(selected_words, 1):
            if not isinstance(word_data, dict):
                print(f"{i}. Invalid word data (not a dict): {word_data}")
                continue
                
            word = word_data.get('word', 'N/A')
            clue = word_data.get('clue', 'N/A')
            reasoning = word_data.get('reasoning', 'N/A')
            
            print(f"{i}. Word: {word.upper() if isinstance(word, str) else word}")
            print(f"   Clue: {clue if isinstance(clue, str) else 'N/A'}")
            print(f"   Reasoning: {reasoning if isinstance(reasoning, str) else 'N/A'}")
            print()
        
        if len(selected_words) == 0:
            print("No words were selected by the LLM.")
            print("This might mean:")
            print("  - The words didn't match the strict selection criteria")
            print("  - The narrative doesn't contain relevant details for these words")
            print("  - The LLM response format was incorrect")
    else:
        print("\nRaw response (JSON parsing failed):")
        print(response)
        print("\nTroubleshooting:")
        print("  - Check if the response contains valid JSON")
        print("  - Verify the response includes 'selected_words' key")
        print("  - The LLM may have failed to follow the output format")
    
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
        start_time = time.time()
        response = query_ollama(full_prompt, model=args.model, ollama_url=args.ollama_url)
        end_time = time.time()
        llm_duration = end_time - start_time
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
    print(f"\n[DEBUG] LLM generation took {llm_duration:.2f} seconds ({llm_duration/60:.2f} minutes)")


if __name__ == "__main__":
    main()
