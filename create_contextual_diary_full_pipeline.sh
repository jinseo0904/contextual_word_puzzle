#!/bin/bash

# 1. SETUP ENVIRONMENT
# Initialize conda for bash
source /home/mhealth-admin/miniconda3/etc/profile.d/conda.sh
# Activate the conda environment
conda activate jinseo
echo "Conda environment activated." >&2

# 2. CONFIGURATION
PHONE_PATH="/home/mhealth-admin/jin/ACAI_test_data/android_phone_usage_anonymized.csv"
HR_PATH="/home/mhealth-admin/jin/ACAI_test_data/garmin_hr_anonymized.csv"
NOISE_PATH="/home/mhealth-admin/jin/ACAI_test_data/pixel_ambient_noise_anonymized.csv"
STEPS_PATH="/home/mhealth-admin/jin/ACAI_test_data/pixel_steps_anonymized.csv"
UEMA_PATH="/home/mhealth-admin/jin/ACAI_test_data/uEMA_anonymized.csv"
JSON_PATH="/home/mhealth-admin/jin/day3_clusters.json"

DAY_LABEL="day 3"
DATE="2024-01-03"
MODEL="gemma3:27b"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_DIR="/home/mhealth-admin/jin/words_with_friends/spelling_bee"
LOG_FILE="$BASE_DIR/logs/spelling_bee_pipeline_${TIMESTAMP}.log"
SPELLING_BEE_PIPELINE="$BASE_DIR/create_spelling_bee_full_pipeline.sh"
GENERATED_JSON_DIR="$BASE_DIR/generated_jsons"
DEFAULT_GAME_JSON="$GENERATED_JSON_DIR/pruned.json"
GAME_JSON_FILE="$DEFAULT_GAME_JSON"
DIARY_OUTPUT="$GENERATED_JSON_DIR/contextual_diary_${TIMESTAMP}.txt"
CLUE_OUTPUT="$GENERATED_JSON_DIR/contextual_clues_${TIMESTAMP}.json"
CLUE_OUTPUT_DIR="$GENERATED_JSON_DIR/finalized_words_and_clue_lists"
VALIDATED_CLUES="$GENERATED_JSON_DIR/validated_clues_${TIMESTAMP}.json"
GENERIC_CLUES="$GENERATED_JSON_DIR/generic_clues_${TIMESTAMP}.json"
FINAL_CLUES_FILE="$CLUE_OUTPUT_DIR/finalized_spelling_bee_words_and_clues_${TIMESTAMP}.json"

echo "Starting Pipeline... Logging to: $LOG_FILE"
mkdir -p "$(dirname "$LOG_FILE")" "$GENERATED_JSON_DIR" "$CLUE_OUTPUT_DIR"

# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================

# STEP A: Generate the Game/Word List (The 7-letter algorithm)
echo "Step A: Generating Spelling Bee Game JSON..." | tee -a "$LOG_FILE"
if [[ -x "$SPELLING_BEE_PIPELINE" ]]; then
    WORD_PIPELINE_OUTPUT=$(bash "$SPELLING_BEE_PIPELINE" 2>&1 | tee -a "$LOG_FILE")
    NEW_GAME_JSON=$(echo "$WORD_PIPELINE_OUTPUT" | sed -n 's/^Final pruned list saved to: //p' | tail -n1 | tr -d '[:space:]')
    if [[ -n "$NEW_GAME_JSON" && -f "$NEW_GAME_JSON" ]]; then
        GAME_JSON_FILE="$NEW_GAME_JSON"
        echo "Using generated pruned list: $GAME_JSON_FILE" | tee -a "$LOG_FILE"
    else
        echo "Warning: Could not determine pruned list path from pipeline output. Falling back to $DEFAULT_GAME_JSON" | tee -a "$LOG_FILE"
    fi
else
    echo "Warning: Spelling bee pipeline script not found at $SPELLING_BEE_PIPELINE. Using default word list at $GAME_JSON_FILE" | tee -a "$LOG_FILE"
fi
echo

# STEP B: The Main Chain
echo "Step B: Running Data -> Narrative -> Selection Pipeline..."

python generate_contextual_diary_prompt_input.py \
    --phone "$PHONE_PATH" \
    --hr "$HR_PATH" \
    --noise "$NOISE_PATH" \
    --steps "$STEPS_PATH" \
    --uema "$UEMA_PATH" \
    --json "$JSON_PATH" \
    --day_label "$DAY_LABEL" \
    --date "$DATE" \
| tee -a "$LOG_FILE" \
| python generate_contextual_diary.py --model "$MODEL" \
| tee "$DIARY_OUTPUT" \
| tee -a "$LOG_FILE" \
| python generate_full_contextual_clues.py --model "$MODEL" --words_file "$GAME_JSON_FILE" \
| tee "$CLUE_OUTPUT" \
| tee -a "$LOG_FILE"

echo "Validating clues..."
python validate_clues.py --model "$MODEL" --diary_file "$DIARY_OUTPUT" < "$CLUE_OUTPUT" > "$VALIDATED_CLUES"

echo "Generating generic fallback clues..."
python generate_generic_clues.py --model "$MODEL" --words_file "$GAME_JSON_FILE" --batch-size 5 \
| tee "$GENERIC_CLUES" \
| tee -a "$LOG_FILE" >/dev/null

echo "Merging validated clues with pruned word list..."
python finalize_words_and_clues.py \
    --pruned-words "$GAME_JSON_FILE" \
    --validated-clues "$VALIDATED_CLUES" \
    --generic-clues "$GENERIC_CLUES" \
    --output "$FINAL_CLUES_FILE"

echo "Finalized clues saved to: $FINAL_CLUES_FILE" | tee -a "$LOG_FILE"

echo "Pipeline Finished. Final Clues are at $FINAL_CLUES_FILE and log at $LOG_FILE"
