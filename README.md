# 🐝 Spelling Bee Game

A web-based implementation of the NYT Spelling Bee game, built with Flask and vanilla JavaScript.

## Features

- **Setup Phase**: Enter a seed word with 7+ distinct letters and choose a center letter
- **Limited Word List**: 30 carefully selected words (25 most common + 5 random) for balanced gameplay
- **Interactive Honeycomb**: Click letters to build words or type directly
- **Word Validation**: Real-time validation against dictionary
- **Scoring System**: 
  - 4-letter words = 1 point
  - 5+ letter words = length in points
  - Pangrams = word points + 7 bonus
- **Rank Progression**: 10 ranks from Beginner to Queen Bee
  - Beginner (0%), Good Start (2%), Moving Up (5%), Good (8%)
  - Solid (15%), Nice (25%), Great (40%), Amazing (50%)
  - Genius (70%), Queen Bee (100%)
- **Visual Progress Bar**: See your progress with animated rank-ups
- **Pangram Detection**: Special highlighting for words that use all letters
- **Word Frequency**: Uses wordfreq library to get word frequencies
- **Definitions**: Shows word definitions from the dictionary
- **Hints Panel**: Two-column layout with comprehensive hints:
  - **Word Grid Matrix**: Shows word count by starting letter and length (like NYT)
  - **Two-Letter List**: Displays all two-letter starting combinations
  - **BINGO**: Identifies the letter with the most words
  - **AI-Generated Creative Hints**: Click to generate cryptic, fun clues using Ollama
  - Hints styled like NYT forum hints with emojis and wordplay
  - **Validation System**: Ensures hints NEVER reveal the actual word
  - Automatically regenerates hints that contain the word
  - Uses fallback hints if generation fails
- **Debug Panel**: Shows all valid words with point values (for testing/cheating 😉)
- **Beautiful UI**: Modern, responsive two-column design with smooth animations

## Installation

1. Make sure you have Python 3 and the required packages:
```bash
pip install flask flask-cors wordfreq requests
```

2. Install and run Ollama with the gpt-oss:20b model:
```bash
# Install Ollama (if not already installed)
curl https://ollama.ai/install.sh | sh

# Pull the model
ollama pull gpt-oss:20b

# Run Ollama (in a separate terminal)
ollama serve
```

3. Ensure the dictionary is set up at:
```
/home/mhealth-admin/jin/words_with_friends/words_database/dictionary_compact.json
```

## Running the Game

1. Start the Flask server:
```bash
cd /home/mhealth-admin/jin/words_with_friends/spelling_bee
python server.py
```

2. Open your browser and navigate to:
```
http://localhost:5005
```

## How to Play

### Setup
1. Enter a seed word with at least 7 distinct letters (e.g., "sunflower", "background")
2. Click "Validate Seed" to check if it's valid
3. Choose one letter to be the center letter (required in all words)
4. Click "Start Game"

### Playing
1. Form words using the available letters
2. All words must:
   - Be at least 4 letters long
   - Use only the available letters
   - Include the center letter
3. Click the honeycomb letters or type directly
4. Click "Submit" or press Enter to check the word
5. Score points and climb the ranks:
   - 4-letter words = 1 point
   - Longer words = their length in points
   - Pangrams = word points + 7 bonus!
6. Try to reach Genius (70%) or Queen Bee (100%)!

### Controls
- **Click letters**: Add letter to current word
- **Delete**: Remove last letter
- **Shuffle**: Rearrange outer letters
- **Submit**: Check if word is valid

### Hints Panel
The right column provides comprehensive hints:

**Word Grid Matrix**: Shows how many words start with each letter and are X letters long
**Two-Letter List**: Shows all two-letter starting combinations (e.g., CH-2, CO-14)
**BINGO**: The letter with the most words
**AI Creative Hints**: Click "Generate Hints" to get AI-powered cryptic clues!

### Hint Generation & Validation
When you click "Generate Hints with AI":
- Ollama generates creative, cryptic hints for each word
- Hints are styled like NYT Spelling Bee forum hints with emojis and wordplay
- **Smart Validation** ensures hints NEVER reveal the word by checking:
  - Direct word appearance
  - Spaced-out letters (like "T E R R Y")
  - Dotted/dashed letters (like "T.E.R.R.Y" or "C-U-T-E")
  - Letter sequences that spell the word
- Bad hints are automatically regenerated (up to 2 attempts)
- Progress is shown in the terminal
- Safe fallback hints are used if generation fails

### Debug Panel
Click "Show All Words" to see all valid words for the current puzzle (useful for testing or if you're stuck!)

## API Endpoints

- `POST /api/validate-seed`: Validate a seed word
- `POST /api/start-game`: Start a new game with seed and center letter
- `POST /api/check-word`: Validate a submitted word
- `POST /api/generate-hints`: Generate creative AI hints with matrix data

## File Structure

```
spelling_bee/
├── server.py          # Flask backend
├── index.html         # Main HTML page
├── styles.css         # Styling
├── spelling_bee.js    # Game logic
└── README.md          # This file
```

## Technologies Used

- **Backend**: Flask, Python 3
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Dictionary**: JSON-based word dictionary with definitions
- **Word Frequency**: wordfreq library
- **Word Indexing**: Bitmask-based efficient lookup

## Credits

Based on the NYT Spelling Bee game concept.

