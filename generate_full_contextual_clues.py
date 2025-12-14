import sys
import argparse
import requests
import json
import re

# ==========================================
# CONTEXTUAL GENERATION PROMPT (v5)
# ==========================================
CLUE_GENERATION_PROMPT = """
### CONTEXT
You are a game engine generating puzzle clues from a user's diary.
The user must fill in the blank "_______" to guess the word.

### INPUT DATA
**Diary:**
{DIARY_SUMMARY}

**Word List:**
{WORD_LIST}

### INSTRUCTION
Select all words that can *factually* describe the diary events.
Write a clue for each using one of the **Accepted Framing Strategies**.

### ACCEPTED FRAMING STRATEGIES
1. **Time/Space Labeling:** Use the word to label *when* or *where* something happened.
   - *Word:* AFTERNOON. *Diary:* User was at library at 2 PM.
   - *Clue:* "You spent a good chunk of this _______ studying at Snell Library."
   - *Word:* AREA. *Diary:* User went to Seaport.
   - *Clue:* "You traveled to the Seaport _______ today."

2. **Logical Association:** Use a word that is physically necessary for an object/action in the diary.
   - *Word:* FARE. *Diary:* User took the subway.
   - *Clue:* "You probably paid _______ today when taking the subway."
   - *Word:* EATER. *Diary:* User passed a restaurant.
   - *Clue:* "You passed Sweet Tomatoes, a likely stop for a hungry _______."

3. **Action Paraphrasing:** Use the word to describe a movement mentioned in the text.
   - *Word:* REENTER. *Diary:* User went back to library.
   - *Clue:* "You went to Snell in the morning, and then you _______-ed it after class."

### BANNED STRATEGIES (Strict)
- **NO IMAGINATION:** Do not say "You imagined an AERATOR" or "You moved like an OTTER."
- **NO FAKE FEELINGS:** Do not say "You felt FEAR" unless the diary explicitly says "I was scared."

### OUTPUT FORMAT
Output ONLY a valid JSON list.
[
  {
    "word": "FARE",
    "clue": "You likely paid _______ today when you took the subway at 11:09 AM.",
    "strategy": "Logical Association",
    "relevance_score": 5
  }
]
"""

def clean_diary_input(raw_text):
    """Removes logging headers."""
    lines = raw_text.splitlines()
    clean_lines = [l for l in lines if not l.startswith("---") and not l.startswith("Step")]
    return "\n".join(clean_lines).strip()

def parse_word_list(json_file_path):
    """Extracts words from the Spelling Bee JSON."""
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        # Extract just the word strings
        valid_words = [item['word'].upper() for item in data.get('words', [])]
        return valid_words
    except Exception as e:
        print(f"Error parsing word list JSON: {e}", file=sys.stderr)
        return []

def extract_json_from_text(text):
    """Finds a JSON list [...] inside a larger string using Regex."""
    try:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return match.group(0)
        return "[]"
    except:
        return "[]"

def query_ollama(model, prompt):
    url = "http://localhost:11434/api/generate"
    
    # NOTE: We removed "format": "json" to stop the model from choking.
    # We will parse the text manually.
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.7 
    }
    
    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        full_response = resp.json()['response']
        
        # Debugging: Print raw response to stderr so you can see what's happening
        # print(f"DEBUG RAW RESPONSE:\n{full_response}\n", file=sys.stderr)
        
        json_str = extract_json_from_text(full_response)
        return json_str
        
    except Exception as e:
        print(f"Error calling Ollama: {e}", file=sys.stderr)
        return "[]"

def batch_words(words, batch_size=5):
    """Split words into batches of specified size."""
    for i in range(0, len(words), batch_size):
        yield words[i:i + batch_size]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-oss:20b')
    parser.add_argument('--words_file', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=5, help='Number of words to process per batch')
    parser.add_argument('input_diary', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    args = parser.parse_args()

    # 1. Read Inputs
    raw_diary = args.input_diary.read()
    diary_text = clean_diary_input(raw_diary)
    word_list = parse_word_list(args.words_file)

    if not diary_text or not word_list:
        print("Error: Missing diary text or word list.", file=sys.stderr)
        return

    # 2. Generate clues in batches
    print(f"--- SELECTING & CLUING ({args.model}) ---", file=sys.stderr)
    print(f"Processing {len(word_list)} words in batches of {args.batch_size}", file=sys.stderr)
    
    all_clues = []
    total_batches = (len(word_list) + args.batch_size - 1) // args.batch_size
    
    for batch_num, word_batch in enumerate(batch_words(word_list, args.batch_size), 1):
        word_list_str = ", ".join(word_batch)
        print(f"Processing batch {batch_num}/{total_batches} ({len(word_batch)} words)", file=sys.stderr)
        
        final_prompt = CLUE_GENERATION_PROMPT.replace("{DIARY_SUMMARY}", diary_text).replace("{WORD_LIST}", word_list_str)
        clues_json_str = query_ollama(args.model, final_prompt)
        
        # Parse the JSON response and add to all_clues
        try:
            batch_clues = json.loads(clues_json_str)
            if isinstance(batch_clues, list):
                all_clues.extend(batch_clues)
            else:
                print(f"Warning: Batch {batch_num} returned non-list result", file=sys.stderr)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON from batch {batch_num}: {e}", file=sys.stderr)
            print(f"Raw response: {clues_json_str[:200]}...", file=sys.stderr)
    
    # 3. Output combined results
    print(json.dumps(all_clues, indent=2))

if __name__ == "__main__":
    main()