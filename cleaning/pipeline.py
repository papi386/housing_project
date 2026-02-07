import pandas as pd
import numpy as np
import re
import os

# -----------------------------
# Basic Cleaning
# -----------------------------
def clean_basic_structure(df):
    """Initial cleaning: drop unused columns and rows without prices."""
    df = df.copy()
    cols_to_drop = ["scraped_at", "url", "bathrooms"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df = df.dropna(subset=['price'])
    return df

# -----------------------------
# Extract Info from Title
# -----------------------------
def extract_info_from_title(df):
    """Extract rooms, property types, and binary features from the title text."""
    df = df.copy()

    # Extract Rooms
    def get_rooms(title):
        if pd.isna(title): return np.nan
        match = re.search(r'[sS]\+?(\d+)', str(title))
        return float(match.group(1)) if match else np.nan

    mask_rooms = df['rooms'].isna()
    df.loc[mask_rooms, 'rooms'] = df.loc[mask_rooms, 'title'].apply(get_rooms)

    # Extract Property Type
    property_keywords = {
        'Appartement': ['appartement', 'app', 'studio', 's1', 's2', 's3', 's4'],
        'Villa': ['villa'],
        'Duplex': ['duplex'],
        'Maison': ['maison'],
        'Immeuble': ['immeuble'],
        'Terrain': ['terrain']
    }

    def get_prop_type(title):
        title_l = str(title).lower()
        for pt, kws in property_keywords.items():
            if any(kw in title_l for kw in kws): return pt
        return np.nan

    mask_prop = df['property_type'].isna()
    df.loc[mask_prop, 'property_type'] = df.loc[mask_prop, 'title'].apply(get_prop_type)

    # Binary features
    features_map = {
        'swimming_pool': ['piscine', 'pool'],
        'garden': ['jardin', 'garden'],
        'terrace': ['terrasse', 'terrace'],
        'garage': ['garage', 'parking', 'sous-sol'],
        'elevator': ['ascenseur'],
        'air_conditioning': ['climatisation', 'clim'],
        'heating': ['chauffage'],
        'equipped_kitchen': ['cuisine équipée', 'cuisine installée'],
        'security': ['sécurité', 'gardien', 'surveillance']
    }

    for col, keywords in features_map.items():
        if col in df.columns:
            pattern = '|'.join(keywords)
            df.loc[df['title'].str.contains(pattern, case=False, na=False), col] = 1

    return df

# -----------------------------
# Map Location to Region
# -----------------------------
def map_to_region(df):
    """Map location strings to broader geographical regions."""
    df = df.copy()

    def get_region(location):
        if pd.isna(location): return "South"
        loc = str(location).lower()

        gt_kw = ['tunis', 'ariana', 'ben arous', 'manouba', 'lac', 'carthage', 'marsa', 'soukra', 'ezzahra', 'rades', 'kram']
        sahel_kw = ['nabeul', 'hammamet', 'sousse', 'monastir', 'mahdia', 'bizerte']
        ne_kw = ['kef', 'béja', 'jendouba', 'tabarka']

        if any(kw in loc for kw in gt_kw): return "Grand Tunis"
        if any(kw in loc for kw in sahel_kw): return "Sahel"
        if any(kw in loc for kw in ne_kw): return "North-East"
        return "South"

    df['Region'] = df['location'].apply(get_region)
    return df

# -----------------------------
# Impute Numerical Columns
# -----------------------------
def impute_missing_numerical(df):
    """Impute area and rooms based on property type medians with fallbacks."""
    df = df.copy()
    df = df.dropna(subset=['property_type'])

    # Special case for Terrain
    df.loc[(df['property_type'] == 'Terrain') & (df['rooms'].isna()), 'rooms'] = 0

    # Median Imputation per type
    for col in ['area_m2', 'rooms']:
        df[col] = df[col].fillna(df.groupby('property_type')[col].transform('median'))
        # Global fallback
        df[col] = df[col].fillna(df[col].median())

    return df

# -----------------------------
# Luxury Feature
# -----------------------------
def create_luxury_feature(df):
    """Generate the binary is_luxury flag based on amenities."""
    df = df.copy()
    luxury_amenities = [
        'doorman', 'cellar', 'heating', 'security', 
        'double_glazing', 'reinforced_door', 'equipped_kitchen', 'air_conditioning'
    ]
    df['amenity_count'] = df[luxury_amenities].sum(axis=1)
    df['is_luxury'] = ((df['swimming_pool'] == 1) | (df['amenity_count'] >= 5)).astype(int)
    df = df.drop(columns=['amenity_count'])
    return df

# -----------------------------
# Full Preprocessing Pipeline
# -----------------------------
def preprocess_data(df):
    df = clean_basic_structure(df)
    df = extract_info_from_title(df)
    df = map_to_region(df)
    df = impute_missing_numerical(df)
    df = create_luxury_feature(df)
    return df

# -----------------------------
# Generic Pipeline Application & Saving
# -----------------------------
def apply_pipeline_and_save(input_csv_path, output_dir):
    """Apply preprocessing and save to a directory."""
    df = pd.read_csv(input_csv_path)
    df_processed = preprocess_data(df)
    
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(input_csv_path)
    save_path = os.path.join(output_dir, filename)
    
    df_processed.to_csv(save_path, index=False)
    print(f"Processed file saved to: {save_path}")
    return df_processed


# -----------------------------
# Example Usage
# -----------------------------
# Train data
train_df = apply_pipeline_and_save("mubawab_1_to_275.csv", "data/train")

# Test data
# test_df = apply_pipeline_and_save("mubawab_test.csv", "data/test")
