import sys
import argparse
import requests
import json
import re

# ==========================================
# GENERIC CLUE PROMPT
# ==========================================
GENERIC_PROMPT = """
### ROLE
You are a Crossword Puzzle Constructor.
Your task is to write **fun, easy-to-medium difficulty** clues for a list of words.

### INPUT WORDS
{WORD_LIST}

### INSTRUCTIONS
For each word in the list, write one clue.
1. **Style:** Mix it up! Use definitions, synonyms, antonyms, or common phrases (fill-in-the-blank).
2. **Difficulty:** Aim for "Monday Morning Crossword" difficulty. Accessible but clever.
3. **Constraint:** Do NOT use the word itself (or variations of it) in the clue.
   - *Bad:* "To run fast." (Target: RUNNING)
   - *Good:* "Participating in a marathon." (Target: RUNNING)

### OUTPUT FORMAT
Output ONLY a JSON list of objects:
[
  {
    "word": "TARGET",
    "clue": "Your clue here."
  }
]
"""

def parse_game_json(file_path):
    """Extracts the list of words from your Spelling Bee JSON format."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Extract just the word strings
        return [item['word'].upper() for item in data.get('words', [])]
    except Exception as e:
        print(f"Error reading game file: {e}", file=sys.stderr)
        return []

def extract_json_from_text(text):
    """Robust JSON extraction using Regex."""
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return match.group(0)
    return None

def check_spoiler(word, clue):
    """Returns True if the word is revealed in the clue."""
    clean_clue = re.sub(r'[^\w\s]', '', clue.lower())
    clean_word = word.lower()
    return clean_word in clean_clue.split()

def query_ollama(model, prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.7 # Higher temp for creativity/variety
    }
    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()['response']
    except Exception as e:
        print(f"Error calling Ollama: {e}", file=sys.stderr)
        return ""

def batch_process(word_list, batch_size=5):
    for i in range(0, len(word_list), batch_size):
        yield word_list[i:i + batch_size]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gemma3:27b')
    parser.add_argument('--words_file', type=str, required=True, help="Path to the Spelling Bee JSON file")
    parser.add_argument('--batch-size', type=int, default=5, help="Number of words per batch (default: 5)")
    args = parser.parse_args()

    # 1. Load Words
    all_words = parse_game_json(args.words_file)
    if not all_words:
        print("No words found.", file=sys.stderr)
        return

    final_clues = []
    
    print(f"--- GENERATING GENERIC CLUES ({len(all_words)} words) ---", file=sys.stderr)

    # 2. Process in Batches
    for batch in batch_process(all_words, batch_size=args.batch_size):
        # Format the list for the prompt
        batch_str = json.dumps(batch)
        final_prompt = GENERIC_PROMPT.replace("{WORD_LIST}", batch_str)
        
        # Call LLM
        raw_response = query_ollama(args.model, final_prompt)
        json_str = extract_json_from_text(raw_response)
        
        if json_str:
            try:
                data = json.loads(json_str)
                # Post-Processing: Spoiler Check
                for item in data:
                    if not check_spoiler(item['word'], item['clue']):
                        final_clues.append(item)
                    else:
                        print(f"Skipped spoiler: {item['word']} -> {item['clue']}", file=sys.stderr)
            except:
                print(f"Failed to parse batch: {batch}", file=sys.stderr)
        else:
            print(f"No JSON found for batch: {batch}", file=sys.stderr)

    # 3. Output Final JSON
    print(json.dumps(final_clues, indent=2))

if __name__ == "__main__":
    main()
