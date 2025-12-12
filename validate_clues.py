import sys
import argparse
import requests
import json
import re

# ==========================================
# PROMPT: THE CRITIC
# ==========================================
VALIDATION_PROMPT = """
### ROLE
You are a strict Puzzle Editor. Validate these game clues against the diary.

### INPUT DATA
**Diary Context:**
{DIARY_TEXT}

**Proposed Clues:**
{CLUES_JSON}

### TASK
1. Read each clue.
2. Assign a score (1-5).
   - 5 = Perfect match to diary facts.
   - 1 = Hallucination (Refers to events not in the diary).
3. **CRITICAL:** Output the result as a valid JSON list.

### OUTPUT FORMAT
[
  {
    "word": "EXAMPLE",
    "clue": "Original clue text...",
    "score": 5,
    "reason": "Accurate."
  }
]
"""

def extract_json_from_text(text):
    """ Tries to find JSON list in text. Returns None if failed. """
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return match.group(0)
    return None

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
    parser.add_argument('input_clues', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    args = parser.parse_args()

    # 1. Load Diary
    try:
        with open(args.diary_file, 'r') as f:
            diary_text = f.read()
    except Exception as e:
        print(f"DEBUG: Error reading diary: {e}", file=sys.stderr)
        return

    # 2. Load Clues (and DEBUG print them)
    clues_str = args.input_clues.read()
    if not clues_str.strip():
        print("DEBUG: Clues input was empty!", file=sys.stderr)
        print("[]")
        return

    print(f"DEBUG: Received Clues Input:\n{clues_str[:200]}...", file=sys.stderr)

    # 3. Validate
    print(f"--- VALIDATING ({args.model}) ---", file=sys.stderr)
    final_prompt = VALIDATION_PROMPT.replace("{DIARY_TEXT}", diary_text).replace("{CLUES_JSON}", clues_str)
    
    raw_response = query_ollama(args.model, final_prompt)
    
    # DEBUG: Print what the LLM actually said
    print(f"DEBUG: Raw LLM Response:\n{raw_response}\n", file=sys.stderr)

    # 4. Parse & Sort
    json_str = extract_json_from_text(raw_response)
    
    if not json_str:
        print("DEBUG: Could not extract JSON from LLM response.", file=sys.stderr)
        # Fallback: If parsing fails, output the original clues but mark them as unchecked
        print(clues_str) 
        return

    try:
        validated_list = json.loads(json_str)
        # Keep everything with score >= 3
        good_clues = [c for c in validated_list if c.get('score', 0) >= 3]
        good_clues.sort(key=lambda x: x['score'], reverse=True)
        
        # If valid list is empty, it means everything was rated low.
        if not good_clues and validated_list:
             print("DEBUG: All clues were rejected (Score < 3). Outputting raw evaluation for review.", file=sys.stderr)
             print(json.dumps(validated_list, indent=2))
        else:
             print(json.dumps(good_clues, indent=2))
        
    except json.JSONDecodeError:
        print("DEBUG: Validated JSON was malformed.", file=sys.stderr)
        print("[]")

if __name__ == "__main__":
    main()