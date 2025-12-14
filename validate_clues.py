import sys
import argparse
import requests
import json
import re

# ==========================================
# CONTEXTUAL VALIDATOR PROMPT (v5)
# ==========================================
VALIDATION_PROMPT = """
### ROLE
You are a Quality Control bot for a diary puzzle.

### INPUT DATA
**Diary:**
{DIARY_TEXT}

**Clues:**
{CLUES_JSON}

### TASK
Rate each clue (1-5) based on **Contextual Validity**.

**SCORE 5 (VALID):**
- **Fact:** "You passed Caffè Nero." (NERO) -> 5
- **Definition:** "You went to Seaport, which is an AREA." -> 5 (True definition).
- **Logical Link:** "You took the subway, so you paid FARE." -> 5 (Logical assumption).
- **Time/Place:** "You were at the library in the AFTERNOON." -> 5 (True time).

**SCORE 1 (INVALID/HALLUCINATION):**
- **Creative Writing:** "You felt light as an OTTER." -> 1 (Hallucination).
- **Fake Objects:** "You imagined an AERATOR." -> 1 (Not in diary).
- **Fake Actions:** "You ate an ENTREE." -> 1 (Diary only says 'passed restaurant').

### OUTPUT FORMAT
Output ONLY a valid JSON list.
[
  {
    "word": "TARGETWORD",
    "clue": "Original clue...",
    "score": 5,
    "reason": "Valid logical link. Subway implies Fare."
  }
]
"""

def extract_json_from_text(text):
    """ Tries to find JSON list in text. Returns None if failed. """
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return match.group(0)
    return None

def batch_clues(clues, batch_size=5):
    """Split clues into batches of specified size."""
    for i in range(0, len(clues), batch_size):
        yield clues[i:i + batch_size]

def query_ollama(model, prompt):
    url = "http://localhost:11434/api/generate"
    payload = { 
        "model": model, 
        "prompt": prompt, 
        "stream": False, 
        "temperature": 0.1 # Very low temp for strict logic
    }
    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()['response']
    except Exception as e:
        print(f"DEBUG: Error calling Ollama: {e}", file=sys.stderr)
        return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-oss:20b')
    parser.add_argument('--diary_file', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=5, help='Number of clues to validate per batch')
    parser.add_argument('input_clues', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    args = parser.parse_args()

    # 1. Load Diary
    try:
        with open(args.diary_file, 'r') as f:
            diary_text = f.read()
    except Exception as e:
        print(f"DEBUG: Error reading diary: {e}", file=sys.stderr)
        return

    # 2. Load Clues
    clues_str = args.input_clues.read()
    if not clues_str.strip():
        print("DEBUG: Clues input was empty!", file=sys.stderr)
        print("[]")
        return

    # Parse the input clues JSON
    try:
        clues_list = json.loads(clues_str)
        if not isinstance(clues_list, list):
            print("DEBUG: Input clues must be a JSON array.", file=sys.stderr)
            print("[]")
            return
    except json.JSONDecodeError as e:
        print(f"DEBUG: Failed to parse input clues JSON: {e}", file=sys.stderr)
        print("[]")
        return

    if not clues_list:
        print("DEBUG: Clues list is empty.", file=sys.stderr)
        print("[]")
        return

    # 3. Validate clues in batches
    print(f"--- VALIDATING ({args.model}) ---", file=sys.stderr)
    print(f"Processing {len(clues_list)} clues in batches of {args.batch_size}", file=sys.stderr)
    
    all_validated = []
    total_batches = (len(clues_list) + args.batch_size - 1) // args.batch_size
    
    for batch_num, clue_batch in enumerate(batch_clues(clues_list, args.batch_size), 1):
        print(f"Processing batch {batch_num}/{total_batches} ({len(clue_batch)} clues)", file=sys.stderr)
        
        batch_clues_str = json.dumps(clue_batch, indent=2)
        final_prompt = VALIDATION_PROMPT.replace("{DIARY_TEXT}", diary_text).replace("{CLUES_JSON}", batch_clues_str)
        
        raw_response = query_ollama(args.model, final_prompt)
        
        # Parse the JSON response and add to all_validated
        json_str = extract_json_from_text(raw_response)
        
        if not json_str:
            print(f"DEBUG: Could not extract JSON from batch {batch_num} response.", file=sys.stderr)
            print(f"DEBUG: Raw response: {raw_response[:200]}...", file=sys.stderr)
            continue
        
        try:
            batch_validated = json.loads(json_str)
            if isinstance(batch_validated, list):
                all_validated.extend(batch_validated)
            else:
                print(f"Warning: Batch {batch_num} returned non-list result", file=sys.stderr)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON from batch {batch_num}: {e}", file=sys.stderr)
            print(f"Raw response: {json_str[:200]}...", file=sys.stderr)

    # 4. Filter & Sort
    if not all_validated:
        print("DEBUG: No validated clues were returned from any batch.", file=sys.stderr)
        print("[]")
        return

    # Keep everything with score >= 3
    good_clues = [c for c in all_validated if c.get('score', 0) >= 3]
    good_clues.sort(key=lambda x: x['score'], reverse=True)
    
    # If valid list is empty, it means everything was rated low.
    if not good_clues and all_validated:
        print("DEBUG: All clues were rejected (Score < 3). Outputting raw evaluation for review.", file=sys.stderr)
        print(json.dumps(all_validated, indent=2))
    else:
        print(json.dumps(good_clues, indent=2))

if __name__ == "__main__":
    main()