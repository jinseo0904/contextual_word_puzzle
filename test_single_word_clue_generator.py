#!/usr/bin/env python3
"""
Dual-mode test script for AI Word Game generation.
Modes:
1. 'generate': Takes a Narrative + Word List -> Creates Clues (using Priority Logic).
2. 'filter':   Takes existing JSON Clues -> Filters/Polishes them (using Editor Logic).
"""

import json
import requests
import argparse
import sys
import time
import re
from typing import Dict, List, Any, Union

# -----------------------------------------------------------------------------
# INPUT LOADING
# -----------------------------------------------------------------------------

def load_text_file(path: str) -> str:
    """Load raw text from a file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error loading text file {path}: {e}", file=sys.stderr)
        sys.exit(1)

def load_word_list(path: str) -> List[str]:
    """Load a list of words from a TXT (one per line) or JSON file."""
    try:
        content = load_text_file(path)
        # Try JSON first
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return [str(w) for w in data]
            elif isinstance(data, dict):
                # Handle {"words": [...]} format
                return [str(w) for w in list(data.values())[0]]
        except json.JSONDecodeError:
            pass
        
        # Fallback to line-by-line
        return [line.strip() for line in content.splitlines() if line.strip()]
    except Exception as e:
        print(f"Error loading word list {path}: {e}", file=sys.stderr)
        sys.exit(1)

def load_json_input(path: str) -> Any:
    """Load JSON input for the filter mode."""
    content = load_text_file(path)
    # Try to find JSON block if mixed with text
    json_match = re.search(r'(\[.*\]|\{.*\})', content, re.DOTALL)
    if json_match:
        content = json_match.group(1)
    
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON input: {e}", file=sys.stderr)
        sys.exit(1)

# -----------------------------------------------------------------------------
# PROMPT BUILDING
# -----------------------------------------------------------------------------

def build_generator_prompt(template_path: str, narrative: str, words: List[str]) -> str:
    """Injects Narrative and Word List into the Generator Prompt."""
    template = load_text_file(template_path)
    
    # Format word list as a string
    words_formatted = ", ".join(words)
    
    # Replace placeholders
    prompt = template.replace("{{NARRATIVE_SUMMARY}}", narrative)
    prompt = prompt.replace("{{WORD_LIST}}", words_formatted)
    
    # Fallback checks if user didn't use double curly braces
    if "{{WORD_LIST}}" in template and "{{WORD_LIST}}" in prompt:
         # If replacement failed (e.g. strict string matching issues), try append
         prompt += f"\n\nNARRATIVE:\n{narrative}\n\nWORDS:\n{words_formatted}"
         
    return prompt

def build_filter_prompt(template_path: str, json_data: Any) -> str:
    """Injects JSON data into the Editor/Filter Prompt."""
    template = load_text_file(template_path)
    json_str = json.dumps(json_data, indent=2)
    
    # Replace standard placeholders
    if "{{JSON_INPUT}}" in template:
        return template.replace("{{JSON_INPUT}}", json_str)
    elif "[Paste the JSON output here]" in template:
        return template.replace("[Paste the JSON output here]", json_str)
    else:
        # Append if no tag found
        return f"{template}\n\n**Input JSON:**\n{json_str}"

# -----------------------------------------------------------------------------
# OLLAMA INTERACTION
# -----------------------------------------------------------------------------

def query_ollama(prompt: str, model: str, url: str) -> str:
    """Send request to Ollama."""
    api_url = f"{url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7, # Slightly creative but grounded
            "num_ctx": 4096     # Ensure enough context for long narratives
        }
    }
    
    print(f"Sending request to Ollama ({model})...")
    try:
        response = requests.post(api_url, json=payload, timeout=600)
        response.raise_for_status()
        return response.json().get('response', '')
    except Exception as e:
        raise RuntimeError(f"Ollama API Error: {e}")

def extract_json_from_response(response: str) -> List[Dict]:
    """Robust JSON extraction from LLM text response."""
    # 1. Try finding a markdown code block
    code_block = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except:
            pass

    # 2. Try finding the first '[' and last ']'
    try:
        start = response.find('[')
        end = response.rfind(']') + 1
        if start != -1 and end != 0:
            return json.loads(response[start:end])
    except:
        pass

    print("Warning: Could not extract valid JSON from response.", file=sys.stderr)
    return []

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='AI Word Game Pipeline Test')
    
    # Mode selection
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Operation mode')
    
    # MODE 1: GENERATE (Narrative + Words -> Clues)
    p_gen = subparsers.add_parser('generate', help='Generate clues from narrative')
    p_gen.add_argument('--narrative', '-n', required=True, help='Path to daily narrative text file')
    p_gen.add_argument('--words', '-w', required=True, help='Path to word list file (txt or json)')
    
    # MODE 2: FILTER (Clues -> Better Clues)
    p_filt = subparsers.add_parser('filter', help='Filter/Edit existing clues')
    p_filt.add_argument('--input', '-i', required=True, help='Path to input JSON clues')

    # Shared arguments
    for p in [p_gen, p_filt]:
        p.add_argument('--prompt', '-p', required=True, help='Path to prompt template')
        p.add_argument('--model', '-m', default='gemma3:27b', help='Ollama model name')
        p.add_argument('--output', '-o', default='output.json', help='Output file path')
        p.add_argument('--url', default='http://localhost:11434', help='Ollama URL')
        p.add_argument('--dry-run', action='store_true', help='Show prompt only')

    args = parser.parse_args()

    # 1. Build Prompt based on Mode
    if args.mode == 'generate':
        print(f"--- GENERATION MODE ---")
        narrative = load_text_file(args.narrative)
        words = load_word_list(args.words)
        print(f"Loaded Narrative: {len(narrative)} chars")
        print(f"Loaded Words: {len(words)} words")
        full_prompt = build_generator_prompt(args.prompt, narrative, words)
        
    elif args.mode == 'filter':
        print(f"--- FILTER MODE ---")
        json_data = load_json_input(args.input)
        print(f"Loaded Input: {len(json_data)} items")
        full_prompt = build_filter_prompt(args.prompt, json_data)

    # 2. Dry Run or Execute
    if args.dry_run:
        print("\n--- PROMPT PREVIEW ---")
        print(full_prompt)
        print("\n[Dry Run] No request sent.")
        return

    # 3. Query LLM
    start_t = time.time()
    raw_response = query_ollama(full_prompt, args.model, args.url)
    duration = time.time() - start_t
    
    print(f"Response received in {duration:.2f}s")

    # 4. Parse and Save
    parsed_data = extract_json_from_response(raw_response)
    
    if parsed_data:
        with open(args.output, 'w') as f:
            json.dump(parsed_data, f, indent=2)
        print(f"Success! Saved {len(parsed_data)} items to {args.output}")
        
        # Simple stats display
        print("\n--- RESULTS SAMPLE ---")
        for item in parsed_data[:3]:
            print(f"Word: {item.get('word')} | Strategy: {item.get('strategy_used') or item.get('bridge_strategy')}")
            print(f"Clue: {item.get('clue')}")
            print("-")
    else:
        print("Error: No valid JSON found in response.")
        with open(args.output + ".raw.txt", 'w') as f:
            f.write(raw_response)
        print(f"Raw response saved to {args.output}.raw.txt")

if __name__ == "__main__":
    main()