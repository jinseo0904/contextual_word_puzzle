#!/bin/bash

# activate the conda virtual environment (change according to your environment)
source activate jinseo
echo "conda environment activated"

# print which environment is activated
echo "activated environment: $CONDA_DEFAULT_ENV"

# configure paths to the test data txt files
TEST_NARRATIVE_PATH="/home/mhealth-admin/jin/words_with_friends/spelling_bee/generated_narrative_test_day3_user002.txt"

# Step 1. Generate the seed word using the narrative
echo "Runnning the command: python /home/mhealth-admin/jin/words_with_friends/spelling_bee/create_contextual_seed_word.py $TEST_NARRATIVE_PATH"

# Capture stdout of the seed word generation, grep out the validated path, and store it
VALIDATED_JSON_PATH=$(python /home/mhealth-admin/jin/words_with_friends/spelling_bee/create_contextual_seed_word.py $TEST_NARRATIVE_PATH --output /home/mhealth-admin/jin/words_with_friends/spelling_bee/generated_jsons/raw_seed_words.json | tee /dev/tty | grep "✓ Saved validated JSON to:" | awk -F': ' '{print $2}' | tr -d '[:space:]')

echo "Path to validated JSON: $VALIDATED_JSON_PATH"
# find all words that can be generated from the seed word
python /home/mhealth-admin/jin/words_with_friends/spelling_bee/generate_spelling_bee_puzzle.py $VALIDATED_JSON_PATH