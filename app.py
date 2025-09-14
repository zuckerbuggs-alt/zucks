
%%writefile app.py
import streamlit as st
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import math
import plotly.graph_objects as go
import io 
import base64
from skyfield.api import EarthSatellite, load, wgs84
import plotly.express as px
from itertools import permutations
from sgp4.api import Satrec, WGS72, jday
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import traceback
# Constants
GEO_ALTITUDE_KM = 35786
ATMOSPHERIC_BOUNDARY_KM = 120
EARTH_RADIUS_KM = 6371
MU_EARTH = 398600.4418  # km^3/s^2
GRAVEYARD_ALTITUDE_KM = GEO_ALTITUDE_KM + 300.0
DEBRIS_MASS_KG = 1000.0
MANEUVER_TIME_SECONDS = 300
# Page config
st.set_page_config(layout="wide", page_title="Space Debris Tracker & Launch Planner")
st.title(" Space Debris Tracker & Graveyard Transfer Planner")
st.caption("Prototype app for visualizing debris orbits, planning launch windows, and estimating transfer delta-v.")
# Function Definitions
def get_tles_from_url(url):
    """Fetch TLE data from URL with error handling"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        st.error(f"Error fetching TLEs from {url}: {str(e)}")
        return ""

def parse_uploaded_file(content):
    """Parse uploaded file content with various formats"""
    lines = content.splitlines()
    parsed_data = []
    
    # Try to detect if it's a CSV format with headers
    is_csv = False
    
    # Check if the first line contains any header keywords and has multiple columns
    if len(lines) > 0:
        header_keywords = ['norad', 'cat_no', 'debris', 'name', 'tle1', 'tle2', 'object', 'satellite', 'id']
        has_keyword = any(keyword in lines[0].lower() for keyword in header_keywords)
        
        # Check if the line can be split into multiple columns (suggesting CSV format)
        if '\t' in lines[0]:
            parts = lines[0].split('\t')
        elif ',' in lines[0]:
            parts = lines[0].split(',')
        else:
            parts = [lines[0]]
        
        has_multiple_columns = len(parts) >= 3
        
        # Consider it CSV if it has header keywords and multiple columns
        is_csv = has_keyword and has_multiple_columns
    
    if is_csv:
        # CSV format with headers - find column indices
        header_line = lines[0].lower()
        
        # Determine delimiter
        if '\t' in lines[0]:
            delimiter = '\t'
        elif ',' in lines[0]:
            delimiter = ','
        else:
            delimiter = None
        
        if delimiter:
            headers = [h.strip().lower() for h in lines[0].split(delimiter)]
            
            # Find column indices
            norad_idx = None
            name_idx = None
            tle1_idx = None
            tle2_idx = None
            
            for i, header in enumerate(headers):
                if 'norad' in header or 'cat_no' in header or 'id' in header:
                    norad_idx = i
                elif 'name' in header or 'debris' in header or 'object' in header:
                    name_idx = i
                elif 'tle1' in header or 'line1' in header:
                    tle1_idx = i
                elif 'tle2' in header or 'line2' in header:
                    tle2_idx = i
            
            # Parse data lines
            for line in lines[1:]:  # Skip header
                if line.strip():
                    parts = line.split(delimiter)
                    if len(parts) >= 4:
                        # Extract values based on found indices
                        norad_id = parts[norad_idx].strip() if norad_idx is not None else ""
                        name = parts[name_idx].strip() if name_idx is not None else ""
                        tle1 = parts[tle1_idx].strip() if tle1_idx is not None else ""
                        tle2 = parts[tle2_idx].strip() if tle2_idx is not None else ""
                        
                        # If we don't have a name, use NORAD ID
                        if not name and norad_id:
                            name = f"NORAD_{norad_id}"
                        
                        # Validate TLE lines
                        if tle1.startswith('1 ') and tle2.startswith('2 ') and len(tle1) == 69 and len(tle2) == 69:
                            parsed_data.append((name, tle1, tle2))
    else:
        # Standard TLE format (name, line1, line2) or (NORAD ID, name, line1, line2)
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
                
            # Check if this is a NORAD ID line (just a number)
            if line.isdigit():
                norad_id = line
                i += 1
                
                # Next line should be the name
                if i < len(lines):
                    name = lines[i].strip()
                    i += 1
                    
                    # Check if we have enough lines for TLE
                    if i + 1 < len(lines):
                        tle1 = lines[i].strip()
                        tle2 = lines[i+1].strip()
                        
                        # Validate TLE lines
                        if tle1.startswith('1 ') and tle2.startswith('2 ') and len(tle1) == 69 and len(tle2) == 69:
                            parsed_data.append((name, tle1, tle2))
                            i += 2
                        else:
                            # Skip invalid TLE
                            i += 1
                    else:
                        # Not enough lines for a complete TLE
                        i += 1
                else:
                    # Not enough lines for a complete TLE
                    i += 1
            # Check if this is a name line (doesn't start with 1 or 2)
            elif not line.startswith('1 ') and not line.startswith('2 '):
                name = line
                i += 1
                
                # Check if we have enough lines for TLE
                if i + 1 < len(lines):
                    tle1 = lines[i].strip()
                    tle2 = lines[i+1].strip()
                    
                    # Validate TLE lines
                    if tle1.startswith('1 ') and tle2.startswith('2 ') and len(tle1) == 69 and len(tle2) == 69:
                        parsed_data.append((name, tle1, tle2))
                        i += 2
                    else:
                        # Skip invalid TLE
                        i += 1
                else:
                    # Not enough lines for a complete TLE
                    i += 1
            else:
                # This might be a TLE line without a name
                if line.startswith('1 ') and i + 1 < len(lines):
                    tle1 = line
                    tle2 = lines[i+1].strip()
                    
                    if tle2.startswith('2 ') and len(tle1) == 69 and len(tle2) == 69:
                        # Extract satellite number from TLE line 1 (positions 2-7)
                        sat_num = tle1[2:7].strip()
                        name = f"SAT_{sat_num}"
                        parsed_data.append((name, tle1, tle2))
                        i += 2
                    else:
                        i += 1
                else:
                    i += 1
    
    return parsed_data
def calculate_orbital_altitude(satrec, jd, fr):
    """Calculate orbital altitude from satellite record"""
    e, r, v = satrec.sgp4(jd, fr)
    if e != 0:
        raise RuntimeError(f'SGP4 error code {e}')
    distance = np.linalg.norm(r)
    return distance - EARTH_RADIUS_KM
def extract_features_from_tle(df):
    """Extract features from TLE data for classification"""
    now = datetime.now(timezone.utc)
    jd_now, fr_now = jday(now.year, now.month, now.day, now.hour, now.minute, now.second)
    features_list = []
    sat_ids = []
    
    for _, row in df.iterrows():
        sid = row['NORAD_CAT_ID']
        line1 = row['TLE_LINE1']
        line2 = row['TLE_LINE2']
        
        try:
            sat = Satrec.twoline2rv(line1, line2)
            altitude = calculate_orbital_altitude(sat, jd_now, fr_now)
            
            # Extract features from TLE line2
            inclination = float(line2[8:16].strip())
            mean_motion = float(line2[52:63].strip())
            features_list.append([altitude, inclination, mean_motion])
            sat_ids.append(sid)
        except Exception as e:
            st.warning(f"Error processing {sid}: {str(e)}")
            continue
    
    return np.array(features_list) if features_list else np.array([]), sat_ids
def train_sample_classifier():
    """Train a sample classifier if no model exists"""
    num_samples = 100
    np.random.seed(42)
    X_dummy = np.random.rand(num_samples, 3) * [40000, 180, 20]
    y_dummy = np.random.choice([0, 1], num_samples)
    X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_dummy, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, 'rf_debris_classifier.joblib')
    return clf
def classify_debris_ml(features, clf):
    """Classify debris using ML model"""
    preds = clf.predict(features)
    return ["Graveyard Orbit Candidate" if p == 0 else "Atmospheric Deorbit Candidate" for p in preds]
def classify_selected_debris(selected_sats_with_tle):
    """Classify selected debris objects"""
    data = []
    for name, l1, l2, sat in selected_sats_with_tle:
        data.append({
            'NORAD_CAT_ID': name,
            'TLE_LINE1': l1,
            'TLE_LINE2': l2
        })
    
    df = pd.DataFrame(data)
    features, sat_ids = extract_features_from_tle(df)
    
    if len(features) == 0:
        return pd.DataFrame()
    
    try:
        clf = joblib.load('rf_debris_classifier.joblib')
    except FileNotFoundError:
        clf = train_sample_classifier()
    
    classifications = classify_debris_ml(features, clf)
    
    results_df = pd.DataFrame({
        'Satellite Name': sat_ids,
        'Classification': classifications,
    })
    
    return results_df.drop_duplicates(subset=['Satellite Name'])
class DebrisObject:
    """Class representing a debris object with TLE data"""
    def __init__(self, name, tle_line1, tle_line2):
        self.name = name
        self.tle_line1 = tle_line1
        self.tle_line2 = tle_line2
        self.satrec = Satrec.twoline2rv(tle_line1, tle_line2)
        self.current_position = None
        self.current_velocity = None
        self.orbital_radius = None
        
    def update_state(self, time):
        """Update position and velocity at given time"""
        jd, fr = jday(time.year, time.month, time.day, time.hour, time.minute, time.second)
        e, r, v = self.satrec.sgp4(jd, fr)
        if e != 0:
            raise RuntimeError(f'SGP4 error code {e}')
        
        self.current_position = np.array(r)
        self.current_velocity = np.array(v)
        self.orbital_radius = np.linalg.norm(self.current_position)
        
    def get_position_at_time(self, time):
        """Get position at a specific time"""
        jd, fr = jday(time.year, time.month, time.day, time.hour, time.minute, time.second)
        e, r, v = self.satrec.sgp4(jd, fr)
        if e != 0:
            raise RuntimeError(f'SGP4 error code {e}')
        return np.array(r)
    
    def get_delta_v_to_graveyard(self):
        """Calculate delta-v required to move to graveyard orbit"""
        r1 = self.orbital_radius
        r2 = EARTH_RADIUS_KM + GRAVEYARD_ALTITUDE_KM
        
        a_transfer = (r1 + r2) / 2
        v1 = np.sqrt(MU_EARTH / r1)
        v_transfer_peri = np.sqrt(MU_EARTH * (2/r1 - 1/a_transfer))
        dv1 = abs(v_transfer_peri - v1)
        
        v2 = np.sqrt(MU_EARTH / r2)
        v_transfer_apo = np.sqrt(MU_EARTH * (2/r2 - 1/a_transfer))
        dv2 = abs(v2 - v_transfer_apo)
        
        return dv1 + dv2
    
    def get_delta_v_to_deorbit(self):
        """Calculate delta-v required to deorbit"""
        r1 = self.orbital_radius
        r2 = EARTH_RADIUS_KM + ATMOSPHERIC_BOUNDARY_KM
        
        a_transfer = (r1 + r2) / 2
        v1 = np.sqrt(MU_EARTH / r1)
        v_transfer_peri = np.sqrt(MU_EARTH * (2/r1 - 1/a_transfer))
        dv1 = abs(v_transfer_peri - v1)
        
        return dv1
class ChaserShuttle:
    """Class representing a chaser shuttle for debris removal"""
    def __init__(self, initial_position, initial_velocity):
        self.position = np.array(initial_position)
        self.velocity = np.array(initial_velocity)
        
    def transfer_to_debris(self, debris, departure_time):
        """Calculate transfer parameters to reach a debris object"""
        target_pos = debris.get_position_at_time(departure_time + timedelta(hours=1))
        
        r1 = np.linalg.norm(self.position)
        r2 = np.linalg.norm(target_pos)
        
        a_transfer = (r1 + r2) / 2
        transfer_time = np.pi * np.sqrt(a_transfer**3 / MU_EARTH)
        
        v1 = np.sqrt(MU_EARTH / r1)
        v_transfer_peri = np.sqrt(MU_EARTH * (2/r1 - 1/a_transfer))
        dv1 = abs(v_transfer_peri - v1)
        
        return {
            'delta_v': dv1,
            'transfer_time': transfer_time,
            'arrival_time': departure_time + timedelta(seconds=transfer_time)
        }
def calculate_transfer_cost(debris1, debris2, departure_time):
    """Calculate cost to transfer between two debris objects"""
    pos1 = debris1.get_position_at_time(departure_time)
    pos2 = debris2.get_position_at_time(departure_time + timedelta(hours=1))
    
    r1 = np.linalg.norm(pos1)
    r2 = np.linalg.norm(pos2)
    
    a_transfer = (r1 + r2) / 2
    transfer_time = np.pi * np.sqrt(a_transfer**3 / MU_EARTH)
    
    v1 = np.sqrt(MU_EARTH / r1)
    v_transfer_peri = np.sqrt(MU_EARTH * (2/r1 - 1/a_transfer))
    dv1 = abs(v_transfer_peri - v1)
    
    v2 = np.sqrt(MU_EARTH / r2)
    v_transfer_apo = np.sqrt(MU_EARTH * (2/r2 - 1/a_transfer))
    dv2 = abs(v2 - v_transfer_apo)
    
    return {
        'delta_v': dv1 + dv2,
        'transfer_time': transfer_time,
        'distance': np.linalg.norm(pos2 - pos1)
    }
def find_optimal_path(debris_objects, chaser, start_time):
    """Find optimal path for chaser shuttle to visit all debris objects"""
    for debris in debris_objects:
        debris.update_state(start_time)
    
    n_debris = len(debris_objects)
    if n_debris > 5:
        raise ValueError("Maximum 5 debris objects supported")
    
    all_permutations = list(permutations(range(n_debris)))
    best_path = None
    best_cost = float('inf')
    best_details = None
    
    for perm in all_permutations:
        total_delta_v = 0.0
        total_time = 0.0
        current_time = start_time
        path_details = []
        
        for i in range(n_debris):
            debris_idx = perm[i]
            debris = debris_objects[debris_idx]
            
            if i == 0:
                transfer = chaser.transfer_to_debris(debris, current_time)
            else:
                prev_debris = debris_objects[perm[i-1]]
                transfer = calculate_transfer_cost(prev_debris, debris, current_time)
            
            total_delta_v += transfer['delta_v']
            total_time += transfer['transfer_time']
            current_time += timedelta(seconds=transfer['transfer_time'])
            
            dv_graveyard = debris.get_delta_v_to_graveyard()
            dv_deorbit = debris.get_delta_v_to_deorbit()
            
            if dv_graveyard < dv_deorbit:
                action = "graveyard"
                action_dv = dv_graveyard
            else:
                action = "deorbit"
                action_dv = dv_deorbit
            
            force = DEBRIS_MASS_KG * (action_dv * 1000) / MANEUVER_TIME_SECONDS
            
            total_delta_v += action_dv
            total_time += MANEUVER_TIME_SECONDS
            current_time += timedelta(seconds=MANEUVER_TIME_SECONDS)
            
            path_details.append({
                'debris_name': debris.name,
                'action': action,
                'transfer_time': transfer['transfer_time'],
                'maneuver_time': MANEUVER_TIME_SECONDS,
                'delta_v': action_dv,
                'force': force,
                'arrival_time': current_time
            })
        
        cost = total_delta_v
        
        if cost < best_cost:
            best_cost = cost
            best_path = perm
            best_details = {
                'path': [debris_objects[idx].name for idx in perm],
                'total_delta_v': total_delta_v,
                'total_time': total_time,
                'details': path_details,
                'cost': cost
            }
    
    return best_details
def calculate_optimal_path(selected_debris):
    """Calculate optimal path for debris removal mission"""
    debris_objects = []
    for name, tle_line1, tle_line2 in selected_debris:
        try:
            debris = DebrisObject(name, tle_line1, tle_line2)
            debris_objects.append(debris)
        except Exception as e:
            st.warning(f"Error creating debris object {name}: {str(e)}")
    
    if len(debris_objects) < 2:
        raise ValueError("Need at least 2 debris objects for path optimization")
    
    parking_orbit_radius = EARTH_RADIUS_KM + 200.0
    parking_velocity = np.sqrt(MU_EARTH / parking_orbit_radius)
    chaser = ChaserShuttle(
        initial_position=[parking_orbit_radius, 0, 0],
        initial_velocity=[0, parking_velocity, 0]
    )
    
    start_time = datetime.now()
    optimal_path = find_optimal_path(debris_objects, chaser, start_time)
    
    return optimal_path
def format_time(seconds):
    """Format time in seconds to hours, minutes, seconds"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"
# Sidebar controls
st.sidebar.header("TLE Input")
tle_mode = st.sidebar.radio(
    "Provide TLEs via:",
    ("Upload TLE file", "Paste TLEs"),
)
tle_text = "" 
show_all_debris = st.sidebar.checkbox("Show all debris from selected catalog", value=False)


if tle_mode == "Upload TLE file":
    up = st.sidebar.file_uploader("Upload TLE file (CSV or text format)", type=["txt", "csv"])
    if up:
        try:
            # Try UTF-8 first
            content = up.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            try:
                # Try Latin-1 as fallback
                content = up.getvalue().decode("latin-1")
            except Exception as e:
                st.error(f"Error decoding file: {str(e)}")
                content = ""
        
        # Parse the uploaded content
        parsed_tles = parse_uploaded_file(content)
        
        # Show debug information
        with st.expander("Debug: Parsed TLEs"):
            st.write(f"Number of TLEs parsed: {len(parsed_tles)}")
            if parsed_tles:
                st.dataframe(pd.DataFrame([(name, l1[:10]+"...", l2[:10]+"...") for name, l1, l2 in parsed_tles], 
                                         columns=["Name", "TLE Line1", "TLE Line2"]))
        
        # Convert parsed TLEs back to text format
        tle_text = ""
        for name, l1, l2 in parsed_tles:
            tle_text += f"{name}\n{l1}\n{l2}\n"
elif tle_mode == "Paste TLEs":
    tle_text = st.sidebar.text_area("Paste TLE text here (name, line1, line2...)", height=200)
# Mission parameters
st.sidebar.header("⚙️ Mission Parameters")
graveyard_alt_km = st.sidebar.number_input(
    "Graveyard altitude (km)",
    value=300.0,
    min_value=50.0,
    max_value=5000.0,
    step=50.0,
)
parking_orbit_km = st.sidebar.number_input("Parking orbit altitude (km)", value=200.0, min_value=100.0, max_value=1000.0, step=10.0)
search_days = st.sidebar.number_input("Search window (days)", min_value=1, max_value=30, value=10)
time_step_seconds = st.sidebar.number_input("Time step (seconds)", min_value=60, max_value=3600, value=300)
proximity_deg = st.sidebar.slider("Ground-track proximity (°)", 0.1, 10.0, 2.0)
max_results = st.sidebar.number_input("Max launch windows", min_value=1, max_value=20, value=6)
# Launch sites
LAUNCH_SITES = [
    {"name": "Kourou (French Guiana)", "lat": 5.23, "lon": -52.77},
    {"name": "Baikonur (Kazakhstan)", "lat": 45.92, "lon": 63.34},
    {"name": "Vandenberg (USA)", "lat": 34.74, "lon": -120.57},
    {"name": "Satish Dhawan (India)", "lat": 13.72, "lon": 80.23},
    {"name": "Cape Canaveral (USA)", "lat": 28.39, "lon": -80.61},
    {"name": "Tanegashima (Japan)", "lat": 30.39, "lon": 130.97},
    {"name": "Jiuquan (China)", "lat": 40.96, "lon": 100.28},
    {"name": "Wallops Flight Facility (USA)", "lat": 37.85, "lon": -75.49},
    {"name": "Kennedy Space Center (USA)", "lat": 28.52, "lon": -80.60},
    {"name": "Plesetsk (Russia)", "lat": 62.92, "lon": 40.57},
    {"name": "Vostochny (Russia)", "lat": 51.88, "lon": 128.32},
    {"name": "Wenchang (China)", "lat": 19.61, "lon": 110.95},
    {"name": "Xichang (China)", "lat": 28.24, "lon": 102.02},
    {"name": "Uchinoura (Japan)", "lat": 31.25, "lon": 131.08},
    {"name": "Semnan (Iran)", "lat": 35.22, "lon": 53.92},
    {"name": "Alcantara (Brazil)", "lat": -2.37, "lon": -44.39},
    {"name": "Pacific Spaceport Complex (USA)", "lat": 57.43, "lon": -156.41},
    {"name": "Mahia Peninsula (New Zealand)", "lat": -39.26, "lon": 177.86},
    {"name": "Palmachim (Israel)", "lat": 31.91, "lon": 34.68},
    {"name": "Sohae (North Korea)", "lat": 39.66, "lon": 124.70},
    {"name": "Taiyuan (China)", "lat": 37.54, "lon": 112.68},
    {"name": "Mid-Atlantic Regional Spaceport (USA)", "lat": 37.83, "lon": -75.49}
]
# Parse TLEs
ts = load.timescale()
sats = []
if tle_text:
    lines = [ln.strip() for ln in tle_text.splitlines() if ln.strip() != ""]
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if line is a name line (not starting with 1 or 2)
        if not line.startswith("1 ") and not line.startswith("2 "):
            # This is a name line
            name = line
            i += 1
            
            # Check if we have enough lines left for TLE
            if i + 1 < len(lines):
                # Get the next two lines
                line1 = lines[i]
                line2 = lines[i+1]
                
                # Basic validation for TLE lines
                if (line1.startswith("1 ") and line2.startswith("2 ") and 
                    len(line1) == 69 and len(line2) == 69):
                    try:
                        sat = EarthSatellite(line1, line2, name, ts)
                        sats.append((name, line1, line2, sat))
                    except Exception as e:
                        st.warning(f"Error creating EarthSatellite for {name}: {str(e)}")
                    
                    i += 2
                else:
                    # Skip invalid TLE
                    i += 1
            else:
                # Not enough lines for a complete TLE
                i += 1
        elif line.startswith("1 "):
            # This is a line1 without a name
            line1 = line
            i += 1
            
            # Check if we have enough lines left for line2
            if i < len(lines):
                line2 = lines[i]
                
                # Basic validation for TLE lines
                if (line2.startswith("2 ") and 
                    len(line1) == 69 and len(line2) == 69):
                    # Extract satellite number from TLE line 1 (positions 2-7)
                    sat_num = line1[2:7].strip()
                    name = f"SAT_{sat_num}"
                    try:
                        sat = EarthSatellite(line1, line2, name, ts)
                        sats.append((name, line1, line2, sat))
                    except Exception as e:
                        st.warning(f"Error creating EarthSatellite for {name}: {str(e)}")
                    
                    i += 1
                else:
                    # Skip invalid TLE
                    i += 1
            else:
                # Not enough lines for a complete TLE
                i += 1
        else:
            # Skip invalid line
            i += 1
# Debug: Show parsed satellites
with st.expander("Debug: Parsed Satellites"):
    st.write(f"Number of satellites parsed: {len(sats)}")
    if sats:
        st.dataframe(pd.DataFrame([(name, l1[:10]+"...", l2[:10]+"...") for name, l1, l2, sat in sats], 
                                 columns=["Name", "TLE Line1", "TLE Line2"]))
if not sats:
    st.warning("No satellites parsed yet. Provide valid TLEs via the sidebar.")
    st.stop()
# Select target satellite
names = [n for n, _, _, _ in sats]
sel_names = st.multiselect(
    "Select up to 5 debris / satellites to track",
    options=names,
    default=names[0] if names else [],
    max_selections=5
)
selected_sats_with_tle = [(n, l1, l2, sat) for n, l1, l2, sat in sats if n in sel_names]
selected_sats = [sat for n, l1, l2, sat in selected_sats_with_tle]
if not selected_sats:
    st.warning("Please select at least one satellite to track.")
    st.stop()
sel_sat = selected_sats[0]
sel_name = sel_sat.name
# Helper functions
R_EARTH = 6378.137
def get_subpoint(satellite, t_sf):
    """Get subpoint (lat, lon, alt) for a satellite at a given time"""
    geoc = satellite.at(t_sf)
    sp = wgs84.subpoint(geoc)
    return sp.latitude.degrees, sp.longitude.degrees, sp.elevation.m / 1000
def hohmann_delta_v(r1, r2):
    """Calculate delta-v for Hohmann transfer"""
    v1, v2 = math.sqrt(MU_EARTH/r1), math.sqrt(MU_EARTH/r2)
    a = 0.5*(r1 + r2)
    v_trans_perigee = math.sqrt(MU_EARTH*(2/r1 - 1/a))
    v_trans_apogee = math.sqrt(MU_EARTH*(2/r2 - 1/a))
    return abs(v_trans_perigee - v1), abs(v2 - v_trans_apogee)
# Current state
now_utc = datetime.now(timezone.utc)
t_now = ts.from_datetime(now_utc)
lat0, lon0, alt0 = get_subpoint(sel_sat, t_now)
r_mag = np.linalg.norm(sel_sat.at(t_now).position.km)
current_alt = r_mag - R_EARTH
st.subheader(f"📡 Current State of {sel_name}")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Latitude (°)", f"{lat0:.2f}")
with col2:
    st.metric("Longitude (°)", f"{lon0:.2f}")
with col3:
    st.metric("Altitude (km)", f"{current_alt:.1f}")
# Delta-v estimate
dv1, dv2 = hohmann_delta_v(r_mag, R_EARTH + graveyard_alt_km)
dv_total = dv1 + dv2
st.markdown("#### Transfer Δv Estimate")
st.write(f"Injection burn: **{dv1:.3f} km/s** | Circularization: **{dv2:.3f} km/s**")
st.success(f"Total Δv (approx): {dv_total:.3f} km/s")
# Search launch windows
st.markdown("---")
st.subheader("Candidate Launch Windows")
search_start, search_end = now_utc, now_utc + timedelta(days=int(search_days))
times = [search_start + timedelta(seconds=i) for i in range(0, int((search_end - search_start).total_seconds()), int(time_step_seconds))]
sf_times = ts.from_datetimes(times)
subpoints = [wgs84.subpoint(sel_sat.at(t)) for t in sf_times]
lats, lons = np.array([sp.latitude.degrees for sp in subpoints]), np.array([sp.longitude.degrees for sp in subpoints])
candidates = []
for site in LAUNCH_SITES:
    lat_diff, lon_diff = np.abs(lats - site["lat"]), np.minimum(np.abs(lons - site["lon"]), 360 - np.abs(lons - site["lon"]))
    idxs = np.where((lat_diff <= proximity_deg) & (lon_diff <= 30))[0]
    for idx in idxs:
        candidates.append({
            "site": site["name"],
            "lat": site["lat"], "lon": site["lon"],
            "utc_time": times[idx],
            "sat_lat": lats[idx], "sat_lon": lons[idx],
            "alt_km": subpoints[idx].elevation.m/1000,
        })
candidates = sorted(candidates, key=lambda x: x["utc_time"])[:max_results]
if candidates:
    df = pd.DataFrame([{
        "Launch Site": c["site"],
        "UTC Time": c["utc_time"].strftime("%Y-%m-%d %H:%M:%S"),
        "Sat Lat": round(c["sat_lat"], 2),
        "Sat Lon": round(c["sat_lon"], 2),
        "Sat Alt (km)": round(c["alt_km"], 1),
    } for c in candidates])
    st.dataframe(df)
else:
    st.info("No candidate launch windows found. Try adjusting search parameters.")
# Map Visualization
df_candidates = pd.DataFrame(candidates)
fig_map = px.scatter_mapbox(
    df_candidates,
    lat="lat",
    lon="lon",
    hover_name="site",
    hover_data={"utc_time": False, "alt_km": True, "lat": False, "lon": False},
    color_discrete_sequence=["red"],
    zoom=1,
    height=400,
    title="Candidate Launch Sites"
)
fig_map.update_layout(
    mapbox_style="open-street-map",
    margin={"r":0,"t":30,"l":0,"b":0}
)
st.plotly_chart(fig_map, use_container_width=True)
# Debris Classification
st.markdown("---")
st.subheader("Debris Classification")
classification_df = classify_selected_debris(selected_sats_with_tle)
if not classification_df.empty:
    selected_names_set = set(sel_names)
    classification_df = classification_df[classification_df['Satellite Name'].isin(selected_names_set)]
    classification_df = classification_df.drop_duplicates(subset=['Satellite Name'])
    
    classification_df['order'] = classification_df['Satellite Name'].apply(lambda x: sel_names.index(x) if x in sel_names else len(sel_names))
    classification_df = classification_df.sort_values('order').drop(columns=['order'])
    
    st.dataframe(classification_df[['Satellite Name', 'Classification']])
    
    st.markdown("""
    **Classification Explanation:**
    - **Graveyard Orbit Candidate**: Debris that should be moved to a higher orbit (graveyard orbit) to prevent interference with operational satellites.
    - **Atmospheric Deorbit Candidate**: Debris that should be deorbited to burn up in Earth's atmosphere.
    """)
else:
    st.info("No debris classification available for selected objects.")
# 3D Orbit Visualization
st.markdown("---")
st.subheader("3D Orbit Visualization")
# Generate Earth sphere data
phi = np.linspace(0, 2 * np.pi, 50)  # Reduced from 100 to 50 for performance
theta = np.linspace(0, np.pi, 50)    # Reduced from 100 to 50 for performance
x_earth = R_EARTH * np.outer(np.cos(phi), np.sin(theta))
y_earth = R_EARTH * np.outer(np.sin(phi), np.sin(theta))
z_earth = R_EARTH * np.outer(np.ones(50), np.cos(theta))
# Create the figure
fig = go.Figure()
# Plot Earth Sphere with texture-like appearance
fig.add_trace(go.Surface(
    x=x_earth, y=y_earth, z=z_earth,
    colorscale=[[0, '#1E88E5'], [0.25, '#43A047'], [0.5, '#FDD835'], [0.75, '#E53935'], [1, '#8E24AA']],
    opacity=0.9,
    showscale=False,
    lighting=dict(ambient=0.2, diffuse=0.8, specular=0.2, roughness=0.5),
    lightposition=dict(x=100, y=100, z=1000)
))
# Sample fewer points for orbit path
sample_rate = max(1, len(times) // 100)  # Limit to ~100 points per orbit
sampled_times = times[::sample_rate]
sampled_sf_times = ts.from_datetimes(sampled_times)
# Define colors for different satellites
colors = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33', '#FF33F3']
# Loop through each selected satellite
for i, sat in enumerate(selected_sats):
    # Generate orbit path in 3D
    positions = sat.at(sampled_sf_times).position.km
    orbit_x, orbit_y, orbit_z = positions[0], positions[1], positions[2]
    
    # Plot the orbit path
    fig.add_trace(go.Scatter3d(
        x=orbit_x, y=orbit_y, z=orbit_z,
        mode='lines',
        line=dict(width=4, color=colors[i % len(colors)]),
        name=f'{sat.name} Orbit'
    ))
    
    # Plot the current satellite position
    current_pos = sat.at(t_now).position.km
    fig.add_trace(go.Scatter3d(
        x=[current_pos[0]], y=[current_pos[1]], z=[current_pos[2]],
        mode='markers',
        marker=dict(size=10, color=colors[i % len(colors)], line=dict(width=2, color='white')),
        name=f'Current Position ({sat.name})'
    ))
# Plot all debris if the checkbox is selected
if show_all_debris:
    all_debris_x, all_debris_y, all_debris_z = [], [], []
    for name, l1, l2, sat in sats:
        if name not in sel_names:
            pos = sat.at(t_now).position.km
            all_debris_x.append(pos[0])
            all_debris_y.append(pos[1])
            all_debris_z.append(pos[2])
    
    fig.add_trace(go.Scatter3d(
        x=all_debris_x, y=all_debris_y, z=all_debris_z,
        mode='markers',
        marker=dict(size=3, color='gray', opacity=0.5),
        name='All Debris'
    ))
# Set the layout with a dark theme
fig.update_layout(
    title_text="3D Orbit Visualization",
    scene=dict(
        xaxis_title='X (km)',
        yaxis_title='Y (km)',
        zaxis_title='Z (km)',
        aspectmode='data',
        bgcolor='rgb(10,10,30)',
        xaxis=dict(backgroundcolor="rgba(0, 0, 0,0)", gridcolor="white", showbackground=True, zerolinecolor="white"),
        yaxis=dict(backgroundcolor="rgba(0, 0, 0,0)", gridcolor="white", showbackground=True, zerolinecolor="white"),
        zaxis=dict(backgroundcolor="rgba(0, 0, 0,0)", gridcolor="white", showbackground=True, zerolinecolor="white")
    ),
    showlegend=True,
    height=700,
    paper_bgcolor='rgb(10,10,30)',
    plot_bgcolor='rgb(10,10,30)',
    font=dict(color='white')
)
st.plotly_chart(fig, use_container_width=True)
# Optimal Path Calculation
st.markdown("---")
st.subheader("Optimal Path Calculation")
if len(selected_sats_with_tle) >= 2:
    selected_debris = [(name, l1, l2) for name, l1, l2, sat in selected_sats_with_tle]
    
    try:
        with st.spinner("Calculating optimal path..."):
            optimal_path = calculate_optimal_path(selected_debris)
        
        # Display optimal path sequence
        st.markdown("### Optimal Debris Sequence")
        path_sequence = " --> ".join(optimal_path['path'])
        st.markdown(f"<h3 style='text-align: center;'>{path_sequence}</h3>", unsafe_allow_html=True)
        
        # Display mission details
        st.markdown("### Mission Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Mission Time", format_time(optimal_path['total_time']))
        with col2:
            st.metric("Total Delta-v", f"{optimal_path['total_delta_v']:.3f} km/s")
        with col3:
            st.metric("Debris Count", len(optimal_path['path']))
        
        # Display debris operations
        st.markdown("### Debris Operations")
        cols = st.columns(len(optimal_path['details']))
        
        for i, (col, detail) in enumerate(zip(cols, optimal_path['details'])):
            with col:
                st.markdown(f"**Step {i+1}**")
                st.markdown(f"##### {detail['debris_name']}")
                st.markdown(f"**Action:** {detail['action'].upper()}")
                st.markdown(f"**Transfer Time:** {format_time(detail['transfer_time'])}")
                st.markdown(f"**Maneuver Time:** {format_time(detail['maneuver_time'])}")
                st.markdown(f"**Delta-v:** {detail['delta_v']:.3f} km/s")
                st.markdown(f"**Force:** {detail['force']:.1f} N")
                st.markdown(f"**Arrival:** {detail['arrival_time'].strftime('%H:%M:%S')}")
                
                if detail['action'] == 'graveyard':
                    st.markdown('<div style="height: 10px; background-color: green; border-radius: 5px;"></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="height: 10px; background-color: red; border-radius: 5px;"></div>', unsafe_allow_html=True)
        
        st.markdown("""
        **Operation Details:**
        - **Transfer Time**: Time required for the chaser shuttle to reach the debris
        - **Maneuver Time**: Time required to push the debris to its destination
        - **Delta-v**: Change in velocity required for the maneuver
        - **Force**: Force required to push the debris (calculated as F = m × a)
        - **Arrival**: Time when the chaser reaches the debris
        
        **Action Indicators:**
        - <div style="display: inline-block; width: 20px; height: 10px; background-color: green; border-radius: 5px;"></div> Graveyard Orbit (move to higher orbit)
        - <div style="display: inline-block; width: 20px; height: 10px; background-color: red; border-radius: 5px;"></div> Atmospheric Deorbit (burn up in atmosphere)
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error calculating optimal path: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
else:
    st.info("Select at least 2 debris objects to calculate the optimal path.")
# Summary
st.markdown("---")
st.subheader("Suggested Plan")
if candidates:
    c0 = candidates[0]
    st.success(f"**Earliest launch window**: {c0['site']} at {c0['utc_time'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
    st.write(f"Satellite altitude: {c0['alt_km']:.1f} km | Lat: {c0['sat_lat']:.2f}°, Lon: {c0['sat_lon']:.2f}°")
    st.write(f"Estimated Δv for graveyard transfer: **{dv_total:.3f} km/s**")
    st.markdown("""
This figure represents the total **change in velocity** ($\Delta v$) required for the transfer mission. This is a direct measure of the energy and fuel needed by the servicing spacecraft. The plan uses a two-burn **Hohmann transfer**, which is the most fuel-efficient method for this maneuver:
* **First Burn**: A prograde burn to boost the spacecraft from the current orbit into an elliptical transfer orbit.
* **Second Burn**: A second burn to circularize the orbit at the higher, designated graveyard altitude.
""")
else:
    st.warning("No valid windows found. Try adjusting search parameters.")
st.caption("This prototype demonstrates key principles of astrodynamics for mission planning.")

