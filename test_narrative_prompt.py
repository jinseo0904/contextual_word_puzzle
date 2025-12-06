#!/usr/bin/env python3
"""
Test script for location narrative generation using Ollama's gemma3:27b model.
Combines the prompt template with actual movement/clustering data.
"""

import csv
import json
import re
import requests
import argparse
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional


def load_prompt_template(prompt_path: str) -> str:
    """Load the prompt template from file."""
    with open(prompt_path, 'r') as f:
        return f.read()


def load_cluster_data(data_path: str) -> Dict[str, Any]:
    """Load the cluster data from JSON file."""
    with open(data_path, 'r') as f:
        return json.load(f)


def format_time(time_str: str) -> str:
    """Format time string to be more readable (e.g., '08:54:00' -> '8:54 AM')."""
    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    return dt.strftime("%I:%M %p").lstrip('0')


def get_cluster_name(cluster_id: int, clusters: List[Dict]) -> str:
    """Get a readable name for a cluster based on nominatim data, including suburb for context."""
    cluster = next((c for c in clusters if int(c['cluster_id']) == cluster_id), None)
    if not cluster:
        return f"Cluster {cluster_id}"
    
    nom = cluster.get('nominatim', {})
    addr = nom.get('address', {})
    
    # Get the primary name
    primary_name = None
    
    # Try to get a meaningful name
    if nom.get('name'):
        primary_name = nom['name']
    # Fall back to address components
    elif addr.get('amenity'):
        primary_name = addr['amenity']
    elif addr.get('building'):
        primary_name = addr['building']
    # Use neighborhood
    elif addr.get('neighbourhood'):
        primary_name = addr['neighbourhood']
    elif addr.get('suburb'):
        primary_name = addr['suburb']
    else:
        primary_name = nom.get('display_name', f"Cluster {cluster_id}").split(',')[0]
    
    # Add suburb in parentheses if available and different from primary name
    suburb = addr.get('suburb')
    if suburb and suburb != primary_name:
        return f"{primary_name} ({suburb})"
    
    return primary_name


def format_pois_for_transition(transition: Dict) -> str:
    """Format POIs for a single transition."""
    pois = transition.get('pois', [])
    if not pois:
        return "No notable POIs along this route."
    
    poi_lines = []
    for poi in pois:
        name = poi.get('name', 'Unknown')
        poi_type = poi.get('type', 'unknown').replace('_', ' ')
        distance = poi.get('distance_m', 0)
        poi_lines.append(f"  - {name} ({poi_type}, {distance:.0f}m from path)")
    
    return '\n'.join(poi_lines)


def generate_poi_data_section(data: Dict) -> str:
    """Generate the POI data section from transitions."""
    transitions = data.get('transitions', [])
    clusters = data.get('clusters', [])
    
    sections = []
    for trans in transitions:
        from_name = get_cluster_name(trans['from_cluster'], clusters)
        to_name = get_cluster_name(trans['to_cluster'], clusters)
        
        section = f"From {from_name} to {to_name}:\n"
        section += format_pois_for_transition(trans)
        sections.append(section)
    
    return '\n\n'.join(sections)


def generate_movement_summary(data: Dict) -> str:
    """Generate the narrative summary section."""
    visits = data.get('visits', [])
    transitions = data.get('transitions', [])
    clusters = data.get('clusters', [])
    
    # Build narrative summary
    summary_parts = []
    summary_parts.append("STAYS:")
    
    for visit in visits:
        cluster_name = get_cluster_name(visit['cluster_id'], clusters)
        start = format_time(visit['start_time'])
        end = format_time(visit['end_time'])
        duration = visit['duration_minutes']
        
        summary_parts.append(
            f"  • {cluster_name}: {start} - {end} ({duration:.0f} minutes)"
        )
    
    summary_parts.append("\nMOVEMENTS:")
    
    for trans in transitions:
        from_name = get_cluster_name(trans['from_cluster'], clusters)
        to_name = get_cluster_name(trans['to_cluster'], clusters)
        departure = format_time(trans['departure_time'])
        arrival = format_time(trans['arrival_time'])
        travel_time = trans['travel_minutes']
        distance = trans['line_distance_m']
        
        summary_parts.append(
            f"  • {from_name} → {to_name}: departed {departure}, "
            f"arrived {arrival} ({travel_time:.0f} min, {distance:.0f}m)"
        )
    
    return '\n'.join(summary_parts)


def load_uema_entries(csv_path: str, day_label: str) -> List[Dict[str, str]]:
    """Load uEMA entries for the specified day."""
    entries: List[Dict[str, str]] = []
    seen = set()
    try:
        with open(csv_path, 'r', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if not row:
                    continue
                day_value = row.get('day', '').strip().lower()
                if day_value != day_label.strip().lower():
                    continue
                time_str = row.get('time_only', '').strip()
                text = row.get('uEMA', '').strip()
                if not text:
                    continue
                key = (time_str, text.lower())
                if key in seen:
                    continue
                seen.add(key)
                entries.append({'time': time_str, 'text': text})
    except FileNotFoundError:
        raise FileNotFoundError(f"uEMA file not found: {csv_path}")
    return entries


def format_uema_section(day_label: str, entries: List[Dict[str, str]]) -> str:
    """Format uEMA entries as a readable section."""
    if not entries:
        return f"No uEMA self-reports available for {day_label}."
    lines = [f"uEMA self-reports for {day_label}:"]
    for entry in entries:
        time_str = entry.get('time', 'Unknown time')
        text = entry.get('text', '')
        lines.append(f"  - {time_str}: {text}")
    return '\n'.join(lines)


def infer_day_from_path(path: str) -> Optional[str]:
    """Infer a day label like 'day 2' from the data path."""
    match = re.search(r"day[\s_-]*(\d+)", path.lower())
    if match:
        return f"day {match.group(1)}"
    return None


def build_full_prompt(template: str, data: Dict, uema_section: str) -> str:
    """Build the complete prompt by replacing placeholders."""
    poi_data = generate_poi_data_section(data)
    movement_summary = generate_movement_summary(data)
    
    # Replace placeholders
    full_prompt = template.replace('<<<POI_DATA>>>', poi_data)
    full_prompt = full_prompt.replace('<<<MOVEMENT_SUMMARY>>>', movement_summary)
    full_prompt = full_prompt.replace('<<<UEMA_RESPONSES>>>', uema_section)
    
    return full_prompt


def query_ollama(prompt: str, model: str = "gemma3:27b") -> str:
    """Send prompt to Ollama and get response."""
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    print(f"Sending request to Ollama ({model})...")
    print("This may take a while...\n")
    
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', '')
    
    except requests.exceptions.RequestException as e:
        return f"Error querying Ollama: {e}"


def main():
    """Main function to test the narrative prompt."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Test location narrative generation with Ollama'
    )
    parser.add_argument(
        '--model', '-m',
        default='gemma3:27b',
        help='Ollama model to use (default: gemma3:27b)'
    )
    parser.add_argument(
        '--prompt',
        default='generate_location_narrative.txt',
        help='Path to prompt template file'
    )
    parser.add_argument(
        '--data',
        default='day3_clusters.json',
        help='Path to cluster data JSON file'
    )
    parser.add_argument(
        '--uema',
        default='ACAI_test_data/uEMA_anonymized.csv',
        help='Path to uEMA CSV file'
    )
    parser.add_argument(
        '--day',
        default=None,
        help="Day label (e.g., 'day 2') for filtering uEMA responses"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only generate and save the prompt, do not call Ollama'
    )
    parser.add_argument(
        '--output',
        default='generated_narrative.txt',
        help='Output file for generated narrative'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Location Narrative Generation Test")
    print("=" * 80)
    print()
    
    # Paths
    prompt_path = args.prompt
    data_path = args.data
    
    # Load data
    print("Loading prompt template...")
    template = load_prompt_template(prompt_path)
    
    print("Loading cluster data...")
    data = load_cluster_data(data_path)
    
    print(f"Found {len(data.get('clusters', []))} clusters")
    print(f"Found {len(data.get('visits', []))} visits")
    print(f"Found {len(data.get('transitions', []))} transitions")
    print()
    
    # Prepare uEMA data (optional)
    day_label: Optional[str] = args.day
    if not day_label:
        day_label = infer_day_from_path(args.data)
        if day_label:
            print(f"Inferred day label '{day_label}' from data path.")

    if day_label:
        print(f"Loading uEMA responses for {day_label}...")
        uema_entries = load_uema_entries(args.uema, day_label)
        uema_section = format_uema_section(day_label, uema_entries)
        print(f"Found {len(uema_entries)} uEMA responses for {day_label}")
    else:
        uema_section = "No uEMA self-reports were provided for this day."
        print("No day specified; skipping uEMA integration.")

    # Build prompt
    print("Building full prompt...")
    full_prompt = build_full_prompt(template, data, uema_section)
    
    # Save the prompt for inspection
    with open('generated_prompt.txt', 'w') as f:
        f.write(full_prompt)
    print("✓ Saved full prompt to 'generated_prompt.txt' for inspection")
    print()
    
    # Show preview of the prompt
    print("=" * 80)
    print("PROMPT PREVIEW (first 1000 chars):")
    print("=" * 80)
    print(full_prompt[:1000])
    print("...\n")
    
    if args.dry_run:
        print("=" * 80)
        print("DRY RUN MODE - Skipping Ollama API call")
        print("=" * 80)
        print("✓ Full prompt saved to 'generated_prompt.txt'")
        print("  You can review it before running with the API.")
        return
    
    # Query Ollama
    print("=" * 80)
    print("QUERYING OLLAMA")
    print("=" * 80)
    print(f"Model: {args.model}")
    response = query_ollama(full_prompt, model=args.model)
    
    # Display response
    print("\n" + "=" * 80)
    print("GENERATED NARRATIVE:")
    print("=" * 80)
    print(response)
    print()
    
    # Save response
    with open(args.output, 'w') as f:
        f.write(response)
    print(f"✓ Saved narrative to '{args.output}'")
    print()
    
    print("=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
