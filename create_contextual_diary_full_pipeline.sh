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
MODEL="gpt-oss:20b"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="/home/mhealth-admin/jin/words_with_friends/spelling_bee/logs/spelling_bee_pipeline_${TIMESTAMP}.log"
GAME_JSON_FILE="/home/mhealth-admin/jin/words_with_friends/spelling_bee/generated_jsons/pruned.json"

echo "Starting Pipeline... Logging to: $LOG_FILE"

# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================

# STEP A: Generate the Game/Word List (The 7-letter algorithm)
# This script should output the JSON structure you showed me to GAME_JSON_FILE
# not implemented yet
# echo "Step A: Generating Spelling Bee Game JSON..."
# python generate_spelling_bee_game.py > "$GAME_JSON_FILE"

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
| \
{
    # Generate Narrative
    python generate_contextual_diary.py --model "$MODEL"
} \
| tee -a "$LOG_FILE" \
| \
{
    # Generate Clues (Reading the narrative from Pipe, Words from the Game JSON)
    python generate_full_contextual_clues.py --model "$MODEL" --words_file "$GAME_JSON_FILE"
} \
>> "$LOG_FILE"

echo "Pipeline Finished. Final Clues are at the bottom of $LOG_FILE"