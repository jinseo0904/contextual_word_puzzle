import pandas as pd
import json
import io
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Update these paths to your actual files
FILES = {
    'phone': '/home/mhealth-admin/jin/ACAI_test_data/android_phone_usage_anonymized.csv',
    'hr': '/home/mhealth-admin/jin/ACAI_test_data/garmin_hr_anonymized.csv',
    'noise': '/home/mhealth-admin/jin/ACAI_test_data/pixel_ambient_noise_anonymized.csv',
    'steps': '/home/mhealth-admin/jin/ACAI_test_data/pixel_steps_anonymized.csv',
    'uema': '/home/mhealth-admin/jin/ACAI_test_data/uEMA_anonymized.csv',
    'json_timeline': '/home/mhealth-admin/jin/day3_clusters.json'
}

# IMPORTANT: The script needs to know which "Day" in the CSV matches your JSON date.
# Based on your data: "day 2" seems to align with the JSON events.
TARGET_DAY_LABEL = "day 3" 
TARGET_CALENDAR_DATE = "2024-01-03" # The date found in your JSON

# ==========================================
# 2. DATA LOADING & PARSING
# ==========================================

def load_sensor_data(filepath):
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip')
        df.columns = [c.lower().strip() for c in df.columns]
        
        if 'day' in df.columns:
            df = df[df['day'] == TARGET_DAY_LABEL].copy()
            
        base_date = datetime.strptime(TARGET_CALENDAR_DATE, "%Y-%m-%d")
        
        def parse_time(row):
            try:
                t = datetime.strptime(row['time_only'], "%I:%M %p")
                return base_date.replace(hour=t.hour, minute=t.minute, second=0)
            except:
                return None

        if 'time_only' in df.columns:
            df['timestamp'] = df.apply(parse_time, axis=1)
            df = df.dropna(subset=['timestamp'])
            
        return df
    except Exception as e:
        # print(f"Warning: Could not read {filepath}. Reason: {e}") # specific error suppression
        return pd.DataFrame()

# Load all sensors
sensors = {name: load_sensor_data(path) for name, path in FILES.items() if name != 'json_timeline'}

# Load JSON Timeline
try:
    with open(FILES['json_timeline'], 'r') as f:
        timeline_data = json.load(f)
except Exception as e:
    print(f"Error loading JSON timeline: {e}")
    timeline_data = {}

# ==========================================
# 3. AUTOMATIC LOCATION NAMING
# ==========================================

def get_location_map(json_data):
    """
    Parses the 'clusters' list to map ID -> "Place Name (Neighborhood)"
    """
    loc_map = {}
    for cluster in json_data.get('clusters', []):
        cid = cluster.get('cluster_id')
        nom = cluster.get('nominatim', {})
        addr = nom.get('address', {})
        
        # 1. Determine Primary Name (Building/Amenity/Neighborhood)
        primary = addr.get('amenity') or \
                  addr.get('building') or \
                  addr.get('shop') or \
                  addr.get('office') or \
                  addr.get('neighbourhood') or \
                  "Unknown Location"
        
        # 2. Determine Secondary Name (Neighborhood/Suburb/City)
        # Avoid repeating the primary name if it's the same
        secondary = addr.get('suburb') or addr.get('city') or ""
        
        if secondary and secondary != primary:
            full_name = f"{primary} ({secondary})"
        else:
            full_name = primary
            
        loc_map[str(cid)] = full_name # Ensure ID is string for lookup
        # Also map integer just in case
        loc_map[int(cid)] = full_name
        
    return loc_map

# Generate the map from the loaded JSON
LOCATION_NAMES = get_location_map(timeline_data)

# ==========================================
# 4. CONTEXT TAGGING LOGIC
# ==========================================

def get_context_tags(start, end, sensor_dict):
    tags = []
    
    # Phone
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
                
    # Noise
    df = sensor_dict.get('noise')
    if df is not None and not df.empty:
        mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
        window = df.loc[mask]
        if not window.empty:
            labels = " ".join(window['ambient_noise'].astype(str).tolist()).lower()
            if "silence" in labels: tags.append("QUIET_ATMOSPHERE")
            if any(x in labels for x in ["speech", "laughter", "voice"]): tags.append("SOCIAL_ATMOSPHERE")
            if any(x in labels for x in ["traffic", "car", "outside"]): tags.append("LOUD_ENVIRONMENT")
            if any(x in labels for x in ["walk", "footsteps"]): tags.append("AUDIBLE_MOVEMENT")

    # Heart Rate
    df = sensor_dict.get('hr')
    if df is not None and not df.empty:
        mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
        window = df.loc[mask]
        if not window.empty:
            avg_bpm = window['heart_rate'].mean()
            if avg_bpm > 105: tags.append("HIGH_EXERTION"); tags.append("RUSHED")
            elif avg_bpm < 65: tags.append("RELAXED_STATE")
    
    # uEMA
    df = sensor_dict.get('uema')
    notes = []
    if df is not None and not df.empty:
        mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
        window = df.loc[mask]
        if not window.empty:
            notes = window['uema'].unique().tolist()
            
    return list(set(tags)), notes

# ==========================================
# 5. GENERATE FINAL OUTPUT
# ==========================================

full_timeline = []

# Process Visits (Stays)
for v in timeline_data.get('visits', []):
    # Lookup the pretty name using our new map
    place_name = LOCATION_NAMES.get(v['cluster_id'], f"Cluster {v['cluster_id']}")
    
    full_timeline.append({
        'type': 'STAY',
        'start': datetime.strptime(v['start_time'], "%Y-%m-%d %H:%M:%S"),
        'end': datetime.strptime(v['end_time'], "%Y-%m-%d %H:%M:%S"),
        'location': place_name,
        'pois': []
    })

# Process Transitions (Movements)
for t in timeline_data.get('transitions', []):
    from_name = LOCATION_NAMES.get(t['from_cluster'], f"Cluster {t['from_cluster']}")
    to_name = LOCATION_NAMES.get(t['to_cluster'], f"Cluster {t['to_cluster']}")
    
    full_timeline.append({
        'type': 'MOVEMENT',
        'start': datetime.strptime(t['departure_time'], "%Y-%m-%d %H:%M:%S"),
        'end': datetime.strptime(t['arrival_time'], "%Y-%m-%d %H:%M:%S"),
        'location': f"{from_name} -> {to_name}",
        'pois': [p['name'] for p in t.get('pois', [])]
    })

full_timeline.sort(key=lambda x: x['start'])

print("### COPY BELOW THIS LINE FOR LLM ###\n")
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