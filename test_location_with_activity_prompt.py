#!/usr/bin/env python3
"""
Test script for location with activity triangulation using Ollama's gemma3:27b model.
Combines the prompt template with location cluster data and activity recognition data.
"""

import csv
import json
import re
import requests
import argparse
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path


def load_prompt_template(prompt_path: str) -> str:
    """Load the prompt template from file."""
    with open(prompt_path, 'r') as f:
        return f.read()


def load_cluster_data(data_path: str) -> Dict[str, Any]:
    """Load the cluster data from JSON file."""
    with open(data_path, 'r') as f:
        return json.load(f)


def load_location_cluster_analysis(analysis_path: str) -> Dict[str, Any]:
    """Load location cluster analysis from text file format."""
    with open(analysis_path, 'r') as f:
        content = f.read()
    
    # Parse visits section
    visits = []
    visits_section = re.search(r'Visits:\s*(.*?)(?=\n\n|\nTransitions:)', content, re.DOTALL)
    if visits_section:
        for line in visits_section.group(1).strip().split('\n'):
            if not line.strip():
                continue
            # Parse: "Five Guys [...] (C0): 2024-01-01 06:28:00 → 2024-01-01 06:55:00 (27.0 min, 28 samples)"
            match = re.match(r'(.+?)\s+\(C(\d+)\):\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+→\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+\(([\d.]+)\s+min', line)
            if match:
                cluster_name = match.group(1).strip()
                cluster_id = int(match.group(2))
                start_time = match.group(3)
                end_time = match.group(4)
                duration = float(match.group(5))
                visits.append({
                    'cluster_id': cluster_id,
                    'cluster_name': cluster_name,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration_minutes': duration
                })
    
    # Parse transitions section
    transitions = []
    transitions_section = re.search(r'Movements:\s*(.*?)(?=\n\n|$)', content, re.DOTALL)
    if transitions_section:
        for line in transitions_section.group(1).strip().split('\n'):
            if not line.strip():
                continue
            # Parse departure/arrival lines
            # "Departed Five Guys [...] (C0) at 7:32 AM and arrived at 70, Station Landing [...] (C1) around 7:36 AM, covering roughly 0.05 km in 4.0 minutes."
            match = re.match(
                r'Departed\s+(.+?)\s+\(C(\d+)\)\s+at\s+(\d{1,2}:\d{2}\s+[AP]M)\s+and\s+arrived\s+at\s+(.+?)\s+\(C(\d+)\)\s+around\s+(\d{1,2}:\d{2}\s+[AP]M),\s+covering\s+roughly\s+([\d.]+)\s+km\s+in\s+([\d.]+)\s+minutes',
                line
            )
            if match:
                from_name = match.group(1).strip()
                from_cluster = int(match.group(2))
                departure_time = match.group(3)
                to_name = match.group(4).strip()
                to_cluster = int(match.group(5))
                arrival_time = match.group(6)
                distance_km = float(match.group(7))
                travel_minutes = float(match.group(8))
                
                # Convert time strings to datetime format for consistency
                # We'll need the date from the visits
                transitions.append({
                    'from_cluster': from_cluster,
                    'from_name': from_name,
                    'to_cluster': to_cluster,
                    'to_name': to_name,
                    'departure_time': departure_time,
                    'arrival_time': arrival_time,
                    'distance_km': distance_km,
                    'travel_minutes': travel_minutes
                })
    
    # Extract date from first visit
    date_str = None
    if visits:
        first_visit = visits[0]
        dt = datetime.strptime(first_visit['start_time'], "%Y-%m-%d %H:%M:%S")
        date_str = dt.strftime("%B %d, %Y")
    
    return {
        'visits': visits,
        'transitions': transitions,
        'date': date_str
    }


def format_time(time_str: str) -> str:
    """Format time string to be more readable (e.g., '08:54:00' -> '8:54 AM')."""
    # Handle both formats: "2024-01-01 08:54:00" and "8:54 AM"
    if 'AM' in time_str or 'PM' in time_str:
        return time_str
    
    try:
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%I:%M %p").lstrip('0')
    except ValueError:
        # Try just time format
        try:
            dt = datetime.strptime(time_str, "%H:%M:%S")
            return dt.strftime("%I:%M %p").lstrip('0')
        except ValueError:
            return time_str


def format_place_clusters(visits: List[Dict]) -> str:
    """Format place clusters and stays section."""
    lines = []
    for visit in visits:
        cluster_name = visit.get('cluster_name', f"Cluster {visit['cluster_id']}")
        cluster_id = visit['cluster_id']
        start = format_time(visit['start_time'])
        end = format_time(visit['end_time'])
        duration = visit['duration_minutes']
        
        lines.append(
            f"From {start} to {end}, the person stayed at {cluster_name} (C{cluster_id}) "
            f"for approximately {duration:.1f} minutes."
        )
    
    return '\n'.join(lines)


def format_movements(transitions: List[Dict]) -> str:
    """Format movements between clusters section."""
    lines = []
    for trans in transitions:
        from_name = trans.get('from_name', f"Cluster {trans['from_cluster']}")
        to_name = trans.get('to_name', f"Cluster {trans['to_cluster']}")
        from_cluster = trans['from_cluster']
        to_cluster = trans['to_cluster']
        departure = trans['departure_time']
        arrival = trans['arrival_time']
        distance = trans['distance_km']
        travel_time = trans['travel_minutes']
        
        lines.append(
            f"Departed {from_name} (C{from_cluster}) at {departure} and arrived at "
            f"{to_name} (C{to_cluster}) around {arrival}, covering roughly {distance:.2f} km "
            f"in {travel_time:.1f} minutes."
        )
    
    return '\n'.join(lines)


def load_activity_segments(activity_path: str) -> str:
    """Load activity segments from text file."""
    with open(activity_path, 'r') as f:
        return f.read().strip()


def build_full_prompt(template: str, data: Dict, activity_segments: str) -> str:
    """Build the complete prompt by replacing placeholders."""
    place_clusters = format_place_clusters(data.get('visits', []))
    movements = format_movements(data.get('transitions', []))
    date = data.get('date', 'Unknown date')
    
    # Replace placeholders
    full_prompt = template.replace('{{DATE}}', date)
    full_prompt = full_prompt.replace('{{PLACE_CLUSTERS}}', place_clusters)
    full_prompt = full_prompt.replace('{{MOVEMENTS}}', movements)
    full_prompt = full_prompt.replace('{{ACTIVITY_SEGMENTS}}', activity_segments)
    
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
    """Main function to test the location with activity triangulation prompt."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Test location with activity triangulation using Ollama'
    )
    parser.add_argument(
        '--model', '-m',
        default='gemma3:27b',
        help='Ollama model to use (default: gemma3:27b)'
    )
    parser.add_argument(
        '--prompt',
        default='ACAI_test_data/jinseo_test_from_peerconnect/location_with_activity_triangulation_prompt.txt',
        help='Path to prompt template file'
    )
    parser.add_argument(
        '--data',
        default=None,
        help='Path to cluster data JSON file (or location cluster analysis text file)'
    )
    parser.add_argument(
        '--analysis',
        default='ACAI_test_data/jinseo_test_from_peerconnect/location_cluster_analysis_day1.txt',
        help='Path to location cluster analysis text file (alternative to --data)'
    )
    parser.add_argument(
        '--activity',
        default='ACAI_test_data/jinseo_test_from_peerconnect/detected_activity_summary.txt',
        help='Path to activity segments text file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only generate and save the prompt, do not call Ollama'
    )
    parser.add_argument(
        '--output',
        default='generated_location_activity_narrative.txt',
        help='Output file for generated narrative'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Location with Activity Triangulation Test")
    print("=" * 80)
    print()
    
    # Load prompt template
    print("Loading prompt template...")
    template = load_prompt_template(args.prompt)
    
    # Load data
    if args.data:
        print(f"Loading cluster data from JSON: {args.data}")
        data = load_cluster_data(args.data)
        # Convert JSON format to expected format
        visits = data.get('visits', [])
        transitions = data.get('transitions', [])
        # Extract date from first visit
        date_str = None
        if visits:
            first_visit = visits[0]
            start_time = first_visit.get('start_time', '')
            try:
                dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
                date_str = dt.strftime("%B %d, %Y")
            except ValueError:
                pass
        
        data = {
            'visits': visits,
            'transitions': transitions,
            'date': date_str
        }
    else:
        print(f"Loading location cluster analysis: {args.analysis}")
        data = load_location_cluster_analysis(args.analysis)
    
    print(f"Found {len(data.get('visits', []))} visits")
    print(f"Found {len(data.get('transitions', []))} transitions")
    if data.get('date'):
        print(f"Date: {data['date']}")
    print()
    
    # Load activity segments
    print(f"Loading activity segments: {args.activity}")
    activity_segments = load_activity_segments(args.activity)
    activity_lines = [line for line in activity_segments.split('\n') if line.strip()]
    print(f"Found {len(activity_lines)} activity segments")
    print()
    
    # Build prompt
    print("Building full prompt...")
    full_prompt = build_full_prompt(template, data, activity_segments)
    
    # Save the prompt for inspection
    with open('generated_location_activity_prompt.txt', 'w') as f:
        f.write(full_prompt)
    print("✓ Saved full prompt to 'generated_location_activity_prompt.txt' for inspection")
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
        print("✓ Full prompt saved to 'generated_location_activity_prompt.txt'")
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
