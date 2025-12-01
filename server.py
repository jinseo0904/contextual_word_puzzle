#!/usr/bin/env python3
"""
Flask server for Spelling Bee game.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import random
import requests
from collections import defaultdict
from pathlib import Path

# Add parent directory to path to import the dictionary functions
sys.path.append(str(Path(__file__).parent.parent / "words_database"))
from index_mask_dictionary import (
    load_words_from_json,
    build_mask_index,
    all_words_for_seed,
    word_to_mask,
    mask_to_letters,
    MIN_WORD_LENGTH,
    DICTIONARY_PATH
)
from wordfreq import word_frequency

app = Flask(__name__)
CORS(app)

# Ollama configuration
OLLAMA_MODEL = "gemma3:27b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Global variables to store dictionary data
mask_index = None
dict_data = None


def initialize_dictionary():
    """Load and index the dictionary on startup."""
    global mask_index, dict_data
    
    print("Loading dictionary...")
    words, dict_data = load_words_from_json(DICTIONARY_PATH)
    print(f"Loaded {len(words)} words")
    
    print("Building mask index...")
    mask_index = build_mask_index(words, min_len=MIN_WORD_LENGTH)
    indexed_count = sum(len(v) for v in mask_index.values())
    print(f"Indexed {indexed_count} words")


@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('.', path)


@app.route('/api/validate-seed', methods=['POST'])
def validate_seed():
    """
    Validate a seed word and return available letters.
    
    Expected JSON: {"seed": "sunflower"}
    Returns: {"valid": true, "letters": "eflnors", "letter_count": 7}
    """
    data = request.json
    seed = data.get('seed', '').strip().lower()
    
    if not seed:
        return jsonify({"error": "Seed word is required"}), 400
    
    if not seed.isalpha():
        return jsonify({"error": "Seed must contain only letters"}), 400
    
    mask = word_to_mask(seed)
    letter_count = mask.bit_count()
    
    if letter_count < 7:
        return jsonify({
            "error": f"Seed must contain at least 7 distinct letters. Current: {letter_count}"
        }), 400
    
    all_letters = mask_to_letters(mask)
    
    # If more than 7 letters, take only the first 7
    if letter_count > 7:
        letters = all_letters[:7]
        return jsonify({
            "valid": True,
            "letters": letters,
            "letter_count": 7,
            "note": f"Seed has {letter_count} distinct letters. Using first 7: {letters.upper()}"
        })
    
    return jsonify({
        "valid": True,
        "letters": all_letters,
        "letter_count": letter_count
    })


@app.route('/api/start-game', methods=['POST'])
def start_game():
    """
    Start a new game with seed and center letter.
    
    Expected JSON: {"seed": "sunflower", "center": "n", "day": "2"}
    Returns: {
        "letters": "eflnors",
        "center": "n",
        "valid_words": [{"word": "sunflower", "frequency": 0.00001, "definition": "..."}],
        "pangrams": ["sunflower"],
        "total_words": 42,
        "selected_day": 2
    }
    """
    data = request.json
    seed = data.get('seed', '').strip().lower()
    center = data.get('center', '').strip().lower()
    selected_day = data.get('day', '')
    
    if not seed or not center:
        return jsonify({"error": "Seed and center letter are required"}), 400
    
    if len(center) != 1 or not center.isalpha():
        return jsonify({"error": "Center must be a single letter"}), 400
    
    # Validate seed and get exactly 7 letters
    mask = word_to_mask(seed)
    all_letters = mask_to_letters(mask)
    
    # Use only first 7 distinct letters if seed has more
    letters = all_letters[:7]
    
    if center not in letters:
        return jsonify({"error": "Center letter must be one of the available letters"}), 400
    
    try:
        # Get all valid words using only the 7 letters
        valid_words = all_words_for_seed(
            seed=letters,  # Use the 7-letter string, not the original seed
            center_letter=center,
            mask_index=mask_index,
            min_len=MIN_WORD_LENGTH
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    # Add frequency and definition to each word
    words_with_data = []
    pangrams = []
    
    # Calculate mask for the 7 letters being used
    game_mask = word_to_mask(letters)
    
    for word in valid_words:
        freq = word_frequency(word, 'en')
        definition = dict_data.get(word, "No definition available")
        
        word_data = {
            "word": word,
            "frequency": freq,
            "definition": definition,
            "is_pangram": word_to_mask(word) == game_mask
        }
        words_with_data.append(word_data)
        
        # Check if it's a pangram (uses all 7 game letters)
        if word_data['is_pangram']:
            pangrams.append(word)
    
    # Sort by frequency (descending)
    words_with_data.sort(key=lambda x: (-x['frequency'], x['word']))
    
    # Check if contextual mode exists - if so, select contextual words first
    day_summary_path = None
    if selected_day:
        day_summary_path = Path(f"/home/mhealth-admin/jin/ACAI_test_data/day{selected_day}_summary.txt")
        if not day_summary_path.exists():
            print(f"⚠ Warning: Day {selected_day} summary not found at {day_summary_path}")
            day_summary_path = None
    
    contextual_words = []
    
    if day_summary_path and day_summary_path.exists():
        print(f"✓ Using Day {selected_day} summary for contextual hints")
        print("Selecting contextually relevant words from full list...")
        contextual_words = select_contextual_words(words_with_data, day_summary_path)
        print(f"Found {len(contextual_words)} contextually relevant words")
    
    # Limit to 30 words with 30-40% contextual if available
    if len(words_with_data) > 30:
        if contextual_words:
            # Aim for 30-40% contextual (9-12 words out of 30)
            target_contextual = min(12, len(contextual_words))
            
            # Get top contextual words by frequency
            contextual_set = set(contextual_words)
            contextual_selected = [w for w in words_with_data if w['word'] in contextual_set][:target_contextual]
            
            # Mark contextual words
            for w in contextual_selected:
                w['is_contextual'] = True
            
            # Get remaining from non-contextual, prioritizing high frequency
            non_contextual = [w for w in words_with_data if w['word'] not in contextual_set]
            remaining_count = 30 - len(contextual_selected)
            
            # Take top by frequency + some random
            top_non_contextual = non_contextual[:max(15, remaining_count - 5)]
            random_non_contextual = random.sample(
                non_contextual[len(top_non_contextual):], 
                min(5, remaining_count - len(top_non_contextual), len(non_contextual) - len(top_non_contextual))
            ) if len(non_contextual) > len(top_non_contextual) else []
            
            # Mark non-contextual words
            for w in top_non_contextual + random_non_contextual:
                w['is_contextual'] = False
            
            words_with_data = contextual_selected + top_non_contextual + random_non_contextual
            
            print(f"Selected {len(contextual_selected)} contextual + {len(top_non_contextual) + len(random_non_contextual)} generic = {len(words_with_data)} total words")
            print(f"Contextual percentage: {len(contextual_selected)/len(words_with_data)*100:.1f}%")
        else:
            # No contextual mode - use original logic
            top_25 = words_with_data[:25]
            remainder = words_with_data[25:]
            random_5 = random.sample(remainder, min(5, len(remainder)))
            words_with_data = top_25 + random_5
            
            # Mark all as non-contextual
            for w in words_with_data:
                w['is_contextual'] = False
        
        # Re-sort by frequency
        words_with_data.sort(key=lambda x: (-x['frequency'], x['word']))
    else:
        # Less than 30 words - mark all as non-contextual
        for w in words_with_data:
            w['is_contextual'] = False
    
    # Calculate maximum possible score
    max_score = 0
    for word_data in words_with_data:
        word = word_data['word']
        word_len = len(word)
        
        # 4-letter = 1 point, 5+ = length in points
        if word_len == 4:
            points = 1
        else:
            points = word_len
        
        # Pangram bonus: +7 points
        if word_data['is_pangram']:
            points += 7
        
        max_score += points
    
    return jsonify({
        "letters": letters,
        "center": center,
        "valid_words": words_with_data,
        "pangrams": pangrams,
        "total_words": len(words_with_data),
        "max_score": max_score,
        "selected_day": selected_day
    })


@app.route('/api/check-word', methods=['POST'])
def check_word():
    """
    Check if a word is valid.
    
    Expected JSON: {"word": "sunflower", "valid_words": [...], "letters": "eflnorsuw", "center": "n"}
    Returns: {"valid": true, "message": "Correct!", "is_pangram": true}
    """
    data = request.json
    word = data.get('word', '').strip().lower()
    valid_words_data = data.get('valid_words', [])
    letters = data.get('letters', '')
    center = data.get('center', '')
    
    # Extract just the words
    valid_words = [w['word'] for w in valid_words_data]
    
    if not word:
        return jsonify({"valid": False, "message": "Please enter a word"}), 400
    
    # Check word only uses available letters
    word_letters = set(word)
    available_letters = set(letters)
    invalid_letters = word_letters - available_letters
    
    if invalid_letters:
        return jsonify({
            "valid": False,
            "message": f"Word contains invalid letters: {', '.join(invalid_letters)}"
        })
    
    # Check word contains center letter
    if center and center not in word:
        return jsonify({
            "valid": False,
            "message": f"Word must contain center letter: {center.upper()}"
        })
    
    if len(word) < MIN_WORD_LENGTH:
        return jsonify({
            "valid": False,
            "message": f"Word must be at least {MIN_WORD_LENGTH} letters"
        })
    
    if word in valid_words:
        # Find the word data
        word_data = next((w for w in valid_words_data if w['word'] == word), None)
        is_pangram = word in data.get('pangrams', [])
        
        # Calculate points
        word_len = len(word)
        if word_len == 4:
            points = 1
        else:
            points = word_len
        
        if is_pangram:
            points += 7
        
        message = f"Pangram! 🎉 +{points} points" if is_pangram else f"Correct! +{points} points"
        
        return jsonify({
            "valid": True,
            "message": message,
            "is_pangram": is_pangram,
            "word_data": word_data,
            "points": points
        })
    else:
        return jsonify({
            "valid": False,
            "message": "Not in word list"
        })


@app.route('/api/generate-hints', methods=['POST'])
def generate_hints():
    """
    Generate creative hints for the word list using Ollama.
    
    Expected JSON: {"words": [{"word": "chignon", "is_pangram": false}, ...], "day": "2"}
    Returns: {"hints": {"CH": ["7) French twist on beehive..."], ...}, "matrix": {...}}
    """
    data = request.json
    words_data = data.get('words', [])
    selected_day = data.get('day', '')
    
    if not words_data:
        return jsonify({"error": "No words provided"}), 400
    
    # Build matrix (letter -> length -> count)
    matrix = defaultdict(lambda: defaultdict(int))
    two_letter_pairs = defaultdict(int)
    
    for word_info in words_data:
        word = word_info['word']
        first_letter = word[0].upper()
        length = len(word)
        matrix[first_letter][length] += 1
        
        # Track two-letter combinations
        if len(word) >= 2:
            two_letter = word[:2].upper()
            two_letter_pairs[two_letter] += 1
    
    # Convert matrix to regular dict for JSON serialization
    matrix_dict = {}
    for letter in sorted(matrix.keys()):
        matrix_dict[letter] = dict(matrix[letter])
    
    # Generate hints using Ollama
    hints_by_prefix = generate_ollama_hints(words_data, two_letter_pairs, selected_day)
    
    # Check if contextual mode is active for the selected day
    if selected_day:
        day_summary_path = Path(f"/home/mhealth-admin/jin/ACAI_test_data/day{selected_day}_summary.txt")
        contextual_mode = day_summary_path.exists()
    else:
        contextual_mode = False
    
    return jsonify({
        "matrix": matrix_dict,
        "two_letter_list": dict(sorted(two_letter_pairs.items())),
        "hints": hints_by_prefix,
        "contextual_mode": contextual_mode
    })


def select_contextual_words_batch(word_batch, day_summary):
    """Select contextually relevant words from a single batch."""
    
    word_list_str = ", ".join(word_batch)
    
    prompt = f"""You are analyzing which words from a puzzle game are meaningfully related to a day's activities.

DAY SUMMARY (brief excerpt):
{day_summary[:3000]}

WORD LIST (this batch):
{word_list_str}

TASK:
From the word list above, select ONLY the words that have a meaningful, grounded connection to the day summary.

SELECTION CRITERIA:
✓ INCLUDE words that relate to:
  - Actual activities mentioned (sitting, walking, coding, homework, studying)
  - Locations or transitions (enter, library, building, location names)
  - Physical states (tired, erect, standing, stationary)
  - Temporal aspects (noon, evening, night, hour)
  - Consistency/inconsistency themes (true, untrue, inconsistent, conflicting)
  - Technology use (phone, computer, keyboard, typing)
  - Movement patterns (runner, walker, stepping, moving)
  - Reported activities or sensor data terms

✗ EXCLUDE words that:
  - Have no connection to the summary
  - Would require inventing events not mentioned
  - Are too abstract or metaphorical with no grounding

Be selective but not overly strict. If a word has ANY reasonable connection, include it.

OUTPUT FORMAT:
Return ONLY a comma-separated list of the selected words, nothing else. No explanations.

Your selection:"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                'model': OLLAMA_MODEL,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.3,
                    'num_ctx': 8192
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            selection = result.get('response', '').strip()
            
            # Parse comma-separated words
            selected_words = [w.strip().lower() for w in selection.split(',') if w.strip()]
            # Filter to only words that are actually in the batch
            valid_selected = [w for w in selected_words if w in word_batch]
            
            return valid_selected
        else:
            return []
            
    except Exception as e:
        print(f"  Batch error: {e}")
        return []


def select_contextual_words(words_data, day_summary_path):
    """Select words that are contextually relevant to the day summary using batched processing."""
    
    with open(day_summary_path, 'r') as f:
        day_summary = f.read()
    
    # Prepare word list
    word_list = [w['word'] for w in words_data]
    total_words = len(word_list)
    
    print(f"Processing {total_words} words in batches of 35...")
    
    # Phase 1: Batch processing (35 words per batch)
    BATCH_SIZE = 35
    all_selected = []
    
    for i in range(0, len(word_list), BATCH_SIZE):
        batch = word_list[i:i+BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(word_list) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"  Batch {batch_num}/{total_batches}: Processing {len(batch)} words...")
        
        selected = select_contextual_words_batch(batch, day_summary)
        all_selected.extend(selected)
        
        print(f"  → Selected {len(selected)} words from this batch")
    
    print(f"\nPhase 1 complete: {len(all_selected)} words selected from {total_words} total")
    
    # Phase 2: If we have too many, prune to top candidates
    target_count = 12  # Target 40% of 30 words
    
    if len(all_selected) > target_count * 2:
        print(f"Phase 2: Pruning {len(all_selected)} words down to ~{target_count}...")
        
        # Run another pruning pass on the selected words
        final_selected = []
        for i in range(0, len(all_selected), BATCH_SIZE):
            batch = all_selected[i:i+BATCH_SIZE]
            selected = select_contextual_words_batch(batch, day_summary)
            final_selected.extend(selected)
        
        all_selected = final_selected
        print(f"Phase 2 complete: Pruned to {len(all_selected)} words")
    
    # Phase 3: Final selection if still too many
    if len(all_selected) > target_count * 3:
        print(f"Phase 3: Final pruning from {len(all_selected)} words...")
        all_selected = select_contextual_words_batch(all_selected[:50], day_summary)
        print(f"Phase 3 complete: Final {len(all_selected)} words")
    
    print(f"\n✓ Final contextual words: {', '.join(all_selected[:15])}{'...' if len(all_selected) > 15 else ''}")
    return all_selected


def generate_ollama_hints(words_data, two_letter_pairs, selected_day):
    """Generate creative hints using Ollama for each two-letter prefix."""
    hints_by_prefix = defaultdict(list)
    
    # Check if we have a day summary for contextual hints
    day_summary_path = None
    if selected_day:
        day_summary_path = Path(f"/home/mhealth-admin/jin/ACAI_test_data/day{selected_day}_summary.txt")
    
    use_contextual = day_summary_path and day_summary_path.exists()
    day_summary = ""
    contextual_word_set = set()
    
    if use_contextual:
        with open(day_summary_path, 'r') as f:
            day_summary = f.read()
        
        # Identify which words in this game are contextual
        # These were already selected during the word limiting phase
        # Mark words that should get contextual hints
        contextual_word_set = set([w['word'] for w in words_data if w.get('is_contextual', False)])
        
        print(f"\n{'='*60}")
        print(f"✓ Found Day {selected_day} summary - using CONTEXTUAL hints!")
        print(f"Contextual words in this puzzle: {len(contextual_word_set)}")
        print(f"{'='*60}\n")
    else:
        if selected_day:
            print(f"\n⚠ Day {selected_day} summary not found, using generic hints\n")
    
    # Group words by two-letter prefix
    words_by_prefix = defaultdict(list)
    for word_info in words_data:
        word = word_info['word']
        if len(word) >= 2:
            prefix = word[:2].upper()
            words_by_prefix[prefix].append(word_info)
    
    print(f"\n{'='*70}")
    print(f"HINT GENERATION MODE: {'CONTEXTUAL (sensor-based)' if use_contextual else 'GENERIC PUZZLE-STYLE'}")
    print(f"Total words: {len(words_data)}")
    print(f"Legend: [CONTEXTUAL] = hints based on daily activities")
    print(f"        [GENERIC] = standard puzzle-style hints")
    print(f"        [FALLBACK] = safe generic hint when generation fails")
    print(f"{'='*70}\n")
    
    # Generate hints for each prefix group
    for prefix in sorted(words_by_prefix.keys()):
        words = words_by_prefix[prefix]
        mode_str = "CONTEXTUAL" if use_contextual else "PUZZLE-STYLE"
        print(f"\n[{prefix}] Generating {mode_str} hints for {len(words)} words...")
        
        for word_info in words:
            word = word_info['word']
            is_pangram = word_info.get('is_pangram', False)
            is_contextual_word = word in contextual_word_set
            
            pangram_indicator = " [PANGRAM]" if is_pangram else ""
            context_indicator = " [CONTEXTUAL]" if is_contextual_word else ""
            
            print(f"  Processing: '{word}'{pangram_indicator}{context_indicator}")
            
            # Generate contextual hint only for pre-selected contextual words
            if use_contextual and is_contextual_word:
                hint = generate_contextual_hint(word, is_pangram, day_summary, word_info)
            else:
                hint = generate_single_hint(word, is_pangram)
            
            length_info = str(len(word))
            hints_by_prefix[prefix].append(f"{length_info}) {hint}")
    
    print(f"\n{'='*70}")
    print(f"✓ Hint generation complete!")
    print(f"Mode: {'CONTEXTUAL' if use_contextual else 'GENERIC PUZZLE-STYLE'}")
    print(f"{'='*70}\n")
    
    return dict(hints_by_prefix)


def generate_single_hint(word, is_pangram):
    """Generate a single creative hint for a word using Ollama."""
    max_attempts = 2
    
    for attempt in range(max_attempts):
        try:
            # Call Ollama API
            prompt = f"""Generate a creative, cryptic hint for the word "{word}" in the style of NYT Spelling Bee forum hints.

CRITICAL RULES - YOU MUST FOLLOW THESE:
1. NEVER write the actual word "{word}" in your hint
2. NEVER spell it out with spaces (like "T E R R Y" or "C U T E")
3. NEVER spell it out with periods/dashes (like "T.E.R.R.Y" or "C-U-T-E")
4. NEVER include the word as part of another word or phrase
5. DO NOT use obvious letter-by-letter spelling patterns
6. Be SUBTLE and INDIRECT - make them think!
7. The hint should make people think and discover the word themselves
{'8. This is a PANGRAM (uses all 7 letters) - make it extra challenging! Add 💎 emoji.' if is_pangram else ''}

Good approaches:
- Describe what the word means without saying it
- Use rhymes, puns, or wordplay
- Reference cultural associations or common phrases
- Use emojis to hint at the concept
- Describe its use or context

GOOD EXAMPLES:
- For "chignon": "French twist on beehive knotted hair at nape of neck💎"
- For "coho": "Silver salmon rhymes w/ Santa's laugh💎"
- For "conch": "Large sea snail whose shell is a trumpet"
- For "noon": "When the sun is highest, time shows same digits 🕛"
- For "gnocchi": "Italian potato dumplings that rhyme with hockey💎"

BAD EXAMPLES (DO NOT DO THIS):
- For "terry": "T E R R Y: towel's loops" ❌ (reveals word)
- For "cute": "C U T E: adorable" ❌ (reveals word)
- For "inch": "I.N.C.H: small unit" ❌ (reveals word)

Generate ONLY the hint text (under 15 words), nothing else:"""

            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'gemma3:27b',
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.8,
                        'top_p': 0.9
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                hint = result.get('response', '').strip()
                # Clean up the hint
                hint = hint.replace('"', '').replace('\n', ' ').strip()
                
                # Validate hint doesn't contain the word
                if validate_hint(hint, word):
                    print(f"✓ [GENERIC PUZZLE] Generated for '{word}': {hint[:50]}...")
                    return hint
                else:
                    print(f"✗ [GENERIC PUZZLE] Hint for '{word}' revealed the word (attempt {attempt+1}): {hint}")
                    print(f"  Regenerating...")
                    continue
            else:
                print(f"   [GENERIC FALLBACK] API error for '{word}'")
                return generate_fallback_hint(word)
                
        except Exception as e:
            print(f"✗ [GENERIC FALLBACK] Error for '{word}': {e}")
            return generate_fallback_hint(word)
    
    # If all attempts failed, use fallback
    print(f"   [GENERIC FALLBACK] All attempts failed for '{word}'")
    return generate_fallback_hint(word)


def validate_hint(hint, word):
    """Check if the hint inappropriately reveals the word."""
    hint_lower = hint.lower()
    word_lower = word.lower()
    
    # Check if word appears directly
    if word_lower in hint_lower:
        return False
    
    # Check if word is spelled out with spaces
    spaced_word = ' '.join(word_lower)
    if spaced_word in hint_lower:
        return False
    
    # Check if word is spelled out with periods
    dotted_word = '.'.join(word_lower)
    if dotted_word in hint_lower:
        return False
    
    # Check if word is spelled out with dashes
    dashed_word = '-'.join(word_lower)
    if dashed_word in hint_lower:
        return False
    
    # Check if all letters appear in sequence (might be too strict, but safe)
    hint_letters_only = ''.join(c for c in hint_lower if c.isalpha())
    if word_lower in hint_letters_only:
        return False
    
    return True


def generate_fallback_hint(word):
    """Generate a safe fallback hint that doesn't reveal the word."""
    fallback_hints = [
        f"{len(word)}-letter word that might surprise you 💎",
        "Think carefully about this one 🤔",
        f"A common {len(word)}-letter word you know",
        "You'll kick yourself when you get this one!",
        "This one requires some creative thinking 💡",
    ]
    import random
    hint = random.choice(fallback_hints)
    print(f"   → Using fallback: '{hint}'")
    return hint


def generate_contextual_hint(word, is_pangram, day_summary, word_info):
    """Generate a contextual hint based on day summary and word relevance."""
    
    # Load contextual prompt template
    prompt_path = Path("/home/mhealth-admin/jin/ACAI_test_data/word_clue_prompt.txt")
    with open(prompt_path, 'r') as f:
        context_prompt = f.read()
    
    # Prepare word information
    definition = word_info.get('definition', 'No definition available')
    frequency = word_info.get('frequency', 0)
    
    prompt = f"""{context_prompt}

DAY-LEVEL ACTIVITY SUMMARY:
{day_summary[:3000]}... [truncated]

WORD TO ANALYZE:
Word: "{word}"
Definition: {definition}

TASK:
Generate a clue for "{word}". This word has been PRE-SELECTED as contextually relevant to the day summary.

REQUIREMENTS FOR CONTEXTUAL CLUE:
1. Reference SPECIFIC activities, times, or locations from the summary
2. Use second person ("you") to make it personal and evocative
3. Include concrete details: "earlier", "in the evening", "at the library", "between buildings", "while coding"
4. Make it feel like someone observed your actual day
5. Be suggestive and vivid, not generic or abstract
6. The clue must NEVER reveal the word directly
7. Keep it under 15 words
{'8. Add 💎 emoji at the end if particularly challenging' if is_pangram else ''}

GOOD EXAMPLES:
- "runner" → "What you resembled while darting across campus earlier."
- "erect" → "The posture you briefly shifted into during that practice moment in the evening."
- "enter" → "Action you performed at least four times between locations."
- "coding" → "Your main reported activity while wrist sensors went wild."
- "library" → "The place you mentioned being at, though location data said otherwise."

BAD EXAMPLES (too vague):
- "Only a small amount of the day's activity was captured." ❌
- "Something you did today." ❌
- "A position." ❌

OUTPUT FORMAT:
Line 1: CONTEXTUAL or GENERIC
Line 2: The clue

Your output:"""

    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'gemma3:27b',
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.8,  # Higher temperature for more creative, evocative hints
                    'num_ctx': 8192
                }
            },
            timeout=45
        )
        
        if response.status_code == 200:
            result = response.json()
            full_response = result.get('response', '').strip()
            
            # Parse the response to extract clue type and hint
            lines = full_response.split('\n')
            clue_type = "UNKNOWN"
            hint = full_response
            
            if len(lines) >= 2:
                first_line = lines[0].strip().upper()
                if first_line in ["CONTEXTUAL", "GENERIC"]:
                    clue_type = first_line
                    hint = '\n'.join(lines[1:]).strip()
            
            # Clean up the hint
            hint = hint.replace('"', '').replace('\n', ' ').strip()
            
            # Validate hint doesn't contain the word
            if validate_hint(hint, word):
                print(f"✓ [{clue_type}] Generated for '{word}': {hint[:50]}...")
                return hint
            else:
                print(f"✗ [{clue_type}] Hint for '{word}' revealed the word, using fallback")
                print(f"   [GENERIC] Using fallback hint")
                return generate_fallback_hint(word)
        else:
            print(f"   [GENERIC] API error, using fallback for '{word}'")
            return generate_fallback_hint(word)
            
    except Exception as e:
        print(f"✗ [GENERIC] Error for '{word}': {e}, using fallback")
        return generate_fallback_hint(word)



if __name__ == '__main__':
    print("Initializing Spelling Bee server...")
    initialize_dictionary()
    print("Server ready!")
    app.run(debug=True, host='0.0.0.0', port=5005)

