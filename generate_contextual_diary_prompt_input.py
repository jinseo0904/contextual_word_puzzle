import pandas as pd
import json
import argparse
import sys
from datetime import datetime

# ==========================================
# 1. ARGUMENT PARSING
# ==========================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate LLM Diary Prompt Input from Sensor Data")
    
    # File Paths
    parser.add_argument('--phone', type=str, default='phone_usage.csv', help='Path to phone usage CSV')
    parser.add_argument('--hr', type=str, default='heart_rate.csv', help='Path to heart rate CSV')
    parser.add_argument('--noise', type=str, default='ambient_noise.csv', help='Path to ambient noise CSV')
    parser.add_argument('--steps', type=str, default='step_counts.csv', help='Path to step counts CSV')
    parser.add_argument('--uema', type=str, default='micro_ema.csv', help='Path to microEMA CSV')
    parser.add_argument('--json', type=str, default='location_clusters.json', help='Path to Location JSON')
    
    # Date Configuration
    parser.add_argument('--day_label', type=str, default='day 2', help='The day label in CSVs to filter by (e.g., "day 2")')
    parser.add_argument('--date', type=str, default='2024-01-03', help='The real calendar date to map times to (YYYY-MM-DD)')

    return parser.parse_args()

# ==========================================
# 2. DATA LOADING LOGIC
# ==========================================
def load_sensor_data(filepath, day_label, target_date_str):
    """
    Reads CSV, filters by day_label, and creates a timestamp column based on target_date_str.
    """
    try:
        # Load CSV (skip bad lines for messy sensors like noise)
        df = pd.read_csv(filepath, on_bad_lines='skip')
        
        # Normalize headers to lowercase and strip whitespace
        df.columns = [c.lower().strip() for c in df.columns]
        
        # 1. Filter by Day (if column exists)
        if 'day' in df.columns:
            df = df[df['day'] == day_label].copy()
            
        # 2. Parse Time
        base_date = datetime.strptime(target_date_str, "%Y-%m-%d")
        
        def parse_time(row):
            try:
                # Expecting format like "02:25 PM"
                t_str = row.get('time_only')
                if not t_str: return None
                t = datetime.strptime(t_str, "%I:%M %p")
                return base_date.replace(hour=t.hour, minute=t.minute, second=0)
            except:
                return None

        if 'time_only' in df.columns:
            df['timestamp'] = df.apply(parse_time, axis=1)
            df = df.dropna(subset=['timestamp'])
            
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Warning: Could not read {filepath}. Reason: {e}")
        return pd.DataFrame()

def get_location_map(json_data):
    """
    Maps Cluster IDs to human-readable names based on Nominatim data.
    """
    loc_map = {}
    for cluster in json_data.get('clusters', []):
        cid = cluster.get('cluster_id')
        nom = cluster.get('nominatim', {})
        addr = nom.get('address', {})
        
        # Prioritize specific names over generic ones
        primary = addr.get('amenity') or \
                  addr.get('building') or \
                  addr.get('shop') or \
                  addr.get('office') or \
                  addr.get('neighbourhood') or \
                  "Unknown Location"
        
        secondary = addr.get('suburb') or addr.get('city') or ""
        
        if secondary and secondary != primary:
            full_name = f"{primary} ({secondary})"
        else:
            full_name = primary
            
        loc_map[cid] = full_name 
    return loc_map

# ==========================================
# 3. CONTEXT TAGGING LOGIC
# ==========================================
def get_context_tags(start, end, sensor_dict):
    tags = []
    
    # --- Phone Usage ---
    df = sensor_dict.get('phone')
    if df is not None and not df.empty:
        mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
        window = df.loc[mask]
        if not window.empty:
            is_active = window['in_use'].astype(str).str.lower() == 'true'
            usage_ratio = is_active.sum() / len(window)
            if usage_ratio > 0.4:
                tags.append("HEAVY_PHONE_USE"); tags.append("DISTRACTED")
            elif usage_ratio < 0.1:
                tags.append("FOCUSED_OBSERVATION")

    # --- Ambient Noise ---
    df = sensor_dict.get('noise')
    if df is not None and not df.empty:
        mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
        window = df.loc[mask]
        if not window.empty:
            labels = " ".join(window['ambient_noise'].astype(str).tolist()).lower()
            if "silence" in labels: tags.append("QUIET_ATMOSPHERE")
            if any(x in labels for x in ["speech", "laughter", "voice", "crowd"]): tags.append("SOCIAL_ATMOSPHERE")
            if any(x in labels for x in ["traffic", "car", "outside", "siren"]): tags.append("LOUD_ENVIRONMENT")
            if any(x in labels for x in ["walk", "footsteps"]): tags.append("AUDIBLE_MOVEMENT")

    # --- Heart Rate ---
    df = sensor_dict.get('hr')
    if df is not None and not df.empty:
        mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
        window = df.loc[mask]
        if not window.empty:
            avg_bpm = window['heart_rate'].mean()
            if avg_bpm > 105: tags.append("HIGH_EXERTION"); tags.append("RUSHED")
            elif avg_bpm < 65: tags.append("RELAXED_STATE")

    # --- uEMA Notes ---
    df = sensor_dict.get('uema')
    notes = []
    if df is not None and not df.empty:
        mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
        window = df.loc[mask]
        if not window.empty:
            notes = window['uema'].unique().tolist()
            
    return list(set(tags)), notes

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    args = parse_arguments()

    # 1. Load Sensors
    print(f"Loading sensors for {args.day_label} mapped to {args.date}...", file=sys.stderr)
    sensors = {
        'phone': load_sensor_data(args.phone, args.day_label, args.date),
        'hr': load_sensor_data(args.hr, args.day_label, args.date),
        'noise': load_sensor_data(args.noise, args.day_label, args.date),
        'steps': load_sensor_data(args.steps, args.day_label, args.date),
        'uema': load_sensor_data(args.uema, args.day_label, args.date)
    }

    # 2. Load JSON
    try:
        with open(args.json, 'r') as f:
            timeline_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {args.json}", file=sys.stderr)
        return

    # 3. Map Locations
    loc_map = get_location_map(timeline_data)

    # 4. Build Timeline
    full_timeline = []

    # Process Visits
    for v in timeline_data.get('visits', []):
        cid = v['cluster_id']
        name = loc_map.get(cid) or loc_map.get(str(cid)) or f"Cluster {cid}"
        
        full_timeline.append({
            'type': 'STAY',
            'start': datetime.strptime(v['start_time'], "%Y-%m-%d %H:%M:%S"),
            'end': datetime.strptime(v['end_time'], "%Y-%m-%d %H:%M:%S"),
            'location': name,
            'pois': []
        })

    # Process Transitions
    for t in timeline_data.get('transitions', []):
        cid_from = t['from_cluster']
        cid_to = t['to_cluster']
        
        name_from = loc_map.get(cid_from) or loc_map.get(str(cid_from)) or f"Cluster {cid_from}"
        name_to = loc_map.get(cid_to) or loc_map.get(str(cid_to)) or f"Cluster {cid_to}"

        full_timeline.append({
            'type': 'MOVEMENT',
            'start': datetime.strptime(t['departure_time'], "%Y-%m-%d %H:%M:%S"),
            'end': datetime.strptime(t['arrival_time'], "%Y-%m-%d %H:%M:%S"),
            'location': f"{name_from} -> {name_to}",
            'pois': [p['name'] for p in t.get('pois', [])]
        })

    # Sort
    full_timeline.sort(key=lambda x: x['start'])

    # 5. Output XML
    # This prints to STDOUT so you can pipe it to a file or clipboard
    for event in full_timeline:
        tags, notes = get_context_tags(event['start'], event['end'], sensors)
        
        print(f"<EVENT>")
        print(f"  <TIME>{event['start'].strftime('%I:%M %p')} - {event['end'].strftime('%I:%M %p')}</TIME>")
        print(f"  <TYPE>{event['type']}</TYPE>")
        print(f"  <LOCATION>{event['location']}</LOCATION>")
        print(f"  <CONTEXT_TAGS>{', '.join(tags)}</CONTEXT_TAGS>")
        print(f"  <UEMA_NOTES>{' | '.join(notes)}</UEMA_NOTES>")
        print(f"  <POIS>{', '.join(event['pois'])}</POIS>")
        print(f"</EVENT>\n")

if __name__ == "__main__":
    main()