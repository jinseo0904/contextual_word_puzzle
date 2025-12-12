import sys
import argparse
import requests
import json

# ==========================================
# 1. THE SYSTEM PROMPT (Template)
# ==========================================
SYSTEM_PROMPT_TEMPLATE = """
### SYSTEM INSTRUCTION
You are a diarist writing a daily journal entry. Your goal is to turn structured sensor data into a natural, first-person narrative. You must be precise with times and locations (for later use in a puzzle game), but descriptive with feelings based on sensor tags.

### INPUT DATA
<movement_summary>
{DATA_INPUT}
</movement_summary>

### INSTRUCTIONS

**STEP 1: DECODE THE CONTEXT TAGS**
Before writing, look at the `<CONTEXT_TAGS>` for each event:
- **HIGH_EXERTION / RUSHED:** Use verbs like "rushed," "hurried," "power-walked," or "ran."
- **RELAXED_STATE:** Use verbs like "strolled," "took my time," or "relaxed."
- **LOUD_ENVIRONMENT:** Describe the area as "bustling," "noisy," "busy," or "chaotic."
- **QUIET_ATMOSPHERE:** Describe the area as "peaceful," "quiet," or "calm."
- **HEAVY_PHONE_USE / DISTRACTED:** Mention that you were "glued to your screen," "scrolling," or "distracted by your device."
- **FOCUSED_OBSERVATION:** Mention that you were "watching the crowds," "taking in the sights," or "people-watching."

**STEP 2: MAXIMIZE POI INTEGRATION**
You must include as many POIs as possible from the `<POIS>` list to serve as game clues.
- Group them naturally (e.g., "I passed a string of spots like Starbucks and Dunkin'").
- Prioritize unique landmarks (e.g., "Boston Wharf Co. Sign") over generic chains.

**STEP 3: NARRATIVE GENERATION**
Write the entry chronologically.
- **Time Anchors:** Start every move/stay with the specific time from the `<TIME>` tag.
- **Weave the Notes:** Incorporate text from `<UEMA_NOTES>` naturally.
- **Location Names:** Use the exact text inside `<LOCATION>`.

### OUTPUT
Write only the final diary entry. Do not write preamble. Start directly with the narrative.
"""

# ==========================================
# 2. OLLAMA API HANDLER
# ==========================================
def query_ollama(model, prompt, stream=True):
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }
    
    try:
        response = requests.post(url, json=payload, stream=stream)
        response.raise_for_status()
        
        print(f"--- GENERATING DIARY ({model}) ---\n")
        
        # Stream the output to console in real-time
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                json_line = json.loads(decoded_line)
                if not json_line.get("done"):
                    print(json_line.get("response", ""), end="", flush=True)
        print("\n\n--- END OF ENTRY ---")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama. Is it running on localhost:11434?")
    except Exception as e:
        print(f"An error occurred: {e}")

# ==========================================
# 3. MAIN LOGIC
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Send XML Sensor Data to Ollama")
    parser.add_argument('--model', type=str, default='mistral', help='Ollama model name (e.g., mistral, llama3, gemma)')
    parser.add_argument('input_file', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help='Input file (default: stdin)')
    
    args = parser.parse_args()

    # Read the XML input (either from file or piped from previous script)
    input_xml = args.input_file.read()
    
    if not input_xml.strip():
        print("Error: No input data received.")
        return

    # Inject XML into the System Prompt
    final_prompt = SYSTEM_PROMPT_TEMPLATE.replace("{DATA_INPUT}", input_xml)

    # Run Generation
    query_ollama(args.model, final_prompt)

if __name__ == "__main__":
    main()