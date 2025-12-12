#!/usr/bin/env python3
"""
Test script for word clue editing/filtering using Ollama.
Takes a JSON file with clues and filters them based on quality criteria.
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


def load_input_json(json_path: str) -> List[Dict[str, Any]]:
    """Load input JSON from file.
    
    Handles various formats:
    - Direct array: [...]
    - Object with 'selected_words' key: {"selected_words": [...]}
    - Object with 'words' key: {"words": [...]}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract the array of clues
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        if 'selected_words' in data:
            return data['selected_words']
        elif 'words' in data:
            return data['words']
        else:
            raise ValueError(f"JSON object doesn't contain 'selected_words' or 'words' key. Found keys: {list(data.keys())}")
    else:
        raise ValueError(f"JSON root must be an array or object, got {type(data)}")


def load_input_txt(txt_path: str) -> List[Dict[str, Any]]:
    """Load and parse JSON from a text file.
    
    Handles cases where the file might contain:
    - Plain JSON
    - JSON in markdown code blocks
    - JSON with extra text
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Try to find JSON in markdown code blocks first
    import re
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)```'
    code_block_match = re.search(code_block_pattern, content, re.DOTALL)
    
    if code_block_match:
        json_str = code_block_match.group(1).strip()
    else:
        # Try to find JSON array or object
        arr_start = content.find('[')
        obj_start = content.find('{')
        
        if arr_start == -1 and obj_start == -1:
            raise ValueError("No JSON array or object found in text file")
        
        if arr_start != -1 and (obj_start == -1 or arr_start < obj_start):
            # Array comes first
            start_idx = arr_start
            open_char = '['
            close_char = ']'
        else:
            # Object comes first
            start_idx = obj_start
            open_char = '{'
            close_char = '}'
        
        # Find matching closing bracket/brace
        bracket_count = 0
        end_idx = start_idx
        for i in range(start_idx, len(content)):
            if content[i] == open_char:
                bracket_count += 1
            elif content[i] == close_char:
                bracket_count -= 1
                if bracket_count == 0:
                    end_idx = i + 1
                    break
        
        if bracket_count != 0:
            raise ValueError(f"Unbalanced {open_char}{close_char} in JSON")
        
        json_str = content[start_idx:end_idx]
    
    # Parse the JSON
    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            if 'selected_words' in data:
                return data['selected_words']
            elif 'words' in data:
                return data['words']
            else:
                raise ValueError(f"JSON object doesn't contain 'selected_words' or 'words' key")
        else:
            raise ValueError(f"JSON root must be an array or object")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from text file: {e}")


def build_full_prompt(template: str, clues_json: List[Dict[str, Any]]) -> str:
    """Build the complete prompt by inserting the JSON clues.
    
    Replaces the placeholder "[Paste the JSON output here]" with the formatted JSON.
    """
    # Format the JSON nicely
    clues_json_str = json.dumps(clues_json, indent=2)
    
    # Replace the placeholder
    if "[Paste the JSON output here]" in template:
        full_prompt = template.replace("[Paste the JSON output here]", clues_json_str)
    elif "<<<JSON_INPUT>>>" in template:
        full_prompt = template.replace("<<<JSON_INPUT>>>", clues_json_str)
    elif "<<>>" in template:
        full_prompt = template.replace("<<>>", clues_json_str, 1)
    else:
        # If no placeholder found, append the JSON at the end
        print("Warning: No placeholder found in template, appending JSON at the end")
        full_prompt = template + "\n\n**Input JSON:**\n" + clues_json_str
    
    return full_prompt


def query_ollama(prompt: str, model: str = "gemma3:27b", ollama_url: str = "http://localhost:11434") -> str:
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
        response = requests.post(url, json=payload, timeout=600)  # 10 minute timeout
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', '')
    
    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"Could not connect to Ollama at {ollama_url}. Make sure Ollama is running.")
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Request to Ollama timed out after 600 seconds.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error querying Ollama: {e}")


def parse_json_response(response: str) -> List[Dict[str, Any]]:
    """Try to extract and parse JSON array from the response.
    
    Handles various response formats:
    - JSON in markdown code blocks (```json ... ``` or ``` ... ```)
    - Plain JSON array
    - JSON with extra text before/after
    """
    import re
    
    if not response or not response.strip():
        print("Warning: Empty response from LLM")
        return None
    
    # Try to find JSON in markdown code blocks first
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)```'
    code_block_match = re.search(code_block_pattern, response, re.DOTALL)
    
    if code_block_match:
        # Extract content from code block
        code_content = code_block_match.group(1).strip()
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
        
        # Handle arrays - return directly
        if isinstance(parsed, list):
            return parsed
        
        # Handle objects - check for common keys
        if isinstance(parsed, dict):
            if 'selected_words' in parsed:
                return parsed['selected_words']
            elif 'words' in parsed:
                return parsed['words']
            else:
                print("Warning: JSON object doesn't contain 'selected_words' or 'words' key")
                print(f"Found keys: {list(parsed.keys())}")
                return None
        
        print(f"Warning: JSON root is not an array or object (type: {type(parsed)})")
        return None
        
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse JSON: {e}")
        print(f"JSON string preview (first 500 chars): {json_str[:500]}...")
        if len(json_str) > 500:
            print(f"... (truncated, total length: {len(json_str)} chars)")
        return None


def display_results(input_clues: List[Dict[str, Any]], response: str, filtered_clues: List[Dict[str, Any]] = None):
    """Display the results in a formatted way."""
    print("=" * 80)
    print("CLUE FILTERING RESULTS:")
    print("=" * 80)
    
    print(f"\nInput: {len(input_clues)} clues")
    print(f"Output: {len(filtered_clues) if filtered_clues else 0} clues")
    
    if filtered_clues:
        removed_count = len(input_clues) - len(filtered_clues)
        print(f"Removed: {removed_count} clues ({removed_count/len(input_clues)*100:.1f}%)")
        
        print(f"\n{'='*80}")
        print("FILTERED CLUES:")
        print("=" * 80)
        
        for i, clue_data in enumerate(filtered_clues, 1):
            if not isinstance(clue_data, dict):
                print(f"{i}. Invalid clue data (not a dict): {clue_data}")
                continue
            
            word = clue_data.get('word', 'N/A')
            clue = clue_data.get('clue', 'N/A')
            bridge_strategy = clue_data.get('bridge_strategy', 'N/A')
            relevance_score = clue_data.get('relevance_score', 'N/A')
            
            print(f"\n{i}. Word: {word.upper() if isinstance(word, str) else word}")
            print(f"   Clue: {clue if isinstance(clue, str) else 'N/A'}")
            if bridge_strategy != 'N/A':
                print(f"   Strategy: {bridge_strategy}")
            if relevance_score != 'N/A':
                print(f"   Relevance: {relevance_score}")
        
        if len(filtered_clues) == 0:
            print("\nNo clues passed the filtering criteria.")
            print("This might mean:")
            print("  - All clues had grammatical issues")
            print("  - All clues were too vague or robotic")
            print("  - The LLM applied very strict filtering")
    else:
        print("\nRaw response (JSON parsing failed):")
        print(response)
        print("\nTroubleshooting:")
        print("  - Check if the response contains valid JSON")
        print("  - Verify the response is a JSON array")
        print("  - The LLM may have failed to follow the output format")
    
    print("\n" + "=" * 80)


def main():
    """Main function to test the clue editor prompt."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Test word clue editing/filtering with Ollama'
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
        default='prompts/gemini_clue_editor.txt',
        help='Path to prompt template file (default: prompts/gemini_clue_editor.txt)'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to input JSON or TXT file containing clues to filter'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only generate and save the prompt, do not call Ollama'
    )
    parser.add_argument(
        '--output',
        default='generated_filtered_clues.json',
        help='Output file for filtered clues (JSON)'
    )
    parser.add_argument(
        '--save-prompt',
        default='generated_clue_editor_prompt.txt',
        help='File to save the full prompt for inspection'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Word Clue Editor Test")
    print("=" * 80)
    print()
    
    # Load inputs
    print(f"Loading prompt template from: {args.prompt}")
    template = load_prompt_template(args.prompt)
    
    print(f"Loading input clues from: {args.input}")
    try:
        if args.input.endswith('.json'):
            input_clues = load_input_json(args.input)
        else:
            input_clues = load_input_txt(args.input)
        print(f"✓ Loaded {len(input_clues)} clues")
    except Exception as e:
        print(f"Error loading input file: {e}", file=sys.stderr)
        sys.exit(1)
    print()
    
    # Build prompt
    print("Building full prompt...")
    full_prompt = build_full_prompt(template, input_clues)
    
    # Save the prompt for inspection
    with open(args.save_prompt, 'w') as f:
        f.write(full_prompt)
    print(f"✓ Saved full prompt to '{args.save_prompt}' for inspection")
    print()
    
    # Show preview
    print("=" * 80)
    print("PROMPT PREVIEW:")
    print("=" * 80)
    print(f"Input clues: {len(input_clues)}")
    print(f"Prompt length: {len(full_prompt)} characters")
    print()
    print("First 500 chars of prompt:")
    print(full_prompt[:500])
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
    filtered_clues = parse_json_response(response)
    
    # Display results
    print()
    display_results(input_clues, response, filtered_clues)
    
    # Save response
    if filtered_clues:
        with open(args.output, 'w') as f:
            json.dump(filtered_clues, f, indent=2)
        print(f"✓ Saved filtered clues to '{args.output}'")
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
