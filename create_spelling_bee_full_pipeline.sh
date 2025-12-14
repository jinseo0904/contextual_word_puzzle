#!/bin/bash
set -euo pipefail

CONDA_ENV="jinseo"
BASE_DIR="/home/mhealth-admin/jin/words_with_friends/spelling_bee"
OUTPUT_DIR="$BASE_DIR/generated_jsons"
LOG_DIR="$BASE_DIR/logs"
TIMESTAMP="$(date +%Y-%m-%d-%H-%M)"
LOG_FILE="$LOG_DIR/spelling_bee_word_list_generation_${TIMESTAMP}.log"

CANDIDATE_SCRIPT="$BASE_DIR/generate_seven_letter_candidates.py"
PUZZLE_SCRIPT="$BASE_DIR/generate_spelling_bee_puzzle.py"
FILTER_SCRIPT="$BASE_DIR/filter_and_prune_words.py"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

source /home/mhealth-admin/miniconda3/etc/profile.d/conda.sh
# Activate the conda environment
conda activate jinseo
echo "Conda environment activated." >&2

exec > >(tee -a "$LOG_FILE") 2>&1

echo "======================================================================"
echo "Spelling Bee Word List Generation Pipeline"
echo "Timestamp: $TIMESTAMP"
echo "Log file: $LOG_FILE"
echo "======================================================================"
echo

echo "[1/4] Generating seven-letter candidate words..."
CANDIDATE_OUTPUT="$(python "$CANDIDATE_SCRIPT")"
echo "$CANDIDATE_OUTPUT"

SEED_WORD="$(echo "$CANDIDATE_OUTPUT" | sed -n 's/^Randomly selected candidate: //p' | tail -n1 | tr -d '[:space:]')"
if [[ -z "$SEED_WORD" ]]; then
  echo "Error: Unable to determine seed word from generate_seven_letter_candidates.py output." >&2
  exit 1
fi
echo "Selected seed word: $SEED_WORD"
echo

echo "[2/4] Generating spelling bee puzzle for seed word '$SEED_WORD'..."
PUZZLE_OUTPUT="$(python "$PUZZLE_SCRIPT" --seed-word "$SEED_WORD")"
echo "$PUZZLE_OUTPUT"

PUZZLE_JSON="$(echo "$PUZZLE_OUTPUT" | sed -n 's/^Saving puzzle to: //p' | tail -n1 | xargs)"
if [[ -z "$PUZZLE_JSON" || ! -f "$PUZZLE_JSON" ]]; then
  echo "Error: Unable to find generated puzzle JSON path." >&2
  exit 1
fi
echo "Puzzle JSON created at: $PUZZLE_JSON"
echo

PRUNED_OUTPUT="$OUTPUT_DIR/pruned_spelling_bee_word_list_${TIMESTAMP}.json"
echo "[3/4] Filtering and pruning words..."
python "$FILTER_SCRIPT" "$PUZZLE_JSON" "$PRUNED_OUTPUT"
echo

echo "[4/4] Pipeline complete."
echo "Final pruned list saved to: $PRUNED_OUTPUT"
echo "Full log stored at: $LOG_FILE"
echo "======================================================================"
