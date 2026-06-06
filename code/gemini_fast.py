import pandas as pd
import re
from difflib import SequenceMatcher

# Load the datasets
tableA = pd.read_csv('./tableA.csv')
tableB = pd.read_csv('./tableB.csv')
test = pd.read_csv('./test.csv')

# Merge test IDs with actual data for comparison
merged = test.merge(tableA, left_on='ltable_id', right_on='id', how='left')
merged = merged.rename(columns=lambda x: x + '_A' if x in tableA.columns and x != 'id' else x)
merged = merged.drop(columns=['id'])

merged = merged.merge(tableB, left_on='rtable_id', right_on='id', how='left')
merged = merged.rename(columns=lambda x: x + '_B' if x in tableB.columns and x != 'id' else x)
merged = merged.drop(columns=['id'])

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove text inside parentheses and brackets (e.g., feat, remix, explicit)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize whitespace
    text = " ".join(text.split())
    return text

def parse_time(time_str):
    if not isinstance(time_str, str):
        return None
    try:
        parts = time_str.split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except:
        return None
    return None

def get_similarity(s1, s2):
    """Calculates string similarity ratio."""
    return SequenceMatcher(None, s1, s2).ratio()

results = []
for index, row in merged.iterrows():
    # 1. Clean and Normalize Text
    artist_a = clean_text(row['Artist_Name_A'])
    artist_b = clean_text(row['Artist_Name_B'])
    
    song_a = clean_text(row['Song_Name_A'])
    song_b = clean_text(row['Song_Name_B'])
    
    # 2. Parse Time
    time_a = parse_time(row['Time_A'])
    time_b = parse_time(row['Time_B'])
    
    # 3. Calculate Scores (Based on Attribute Weights)
    # Weights: Song (60%), Time (25%), Artist (15%)
    song_sim = get_similarity(song_a, song_b)
    artist_sim = get_similarity(artist_a, artist_b)
    
    time_diff = abs(time_a - time_b) if (time_a is not None and time_b is not None) else 100
    time_score = 1.0 if time_diff <= 10 else 0.0
    
    # Combined weighted score
    total_score = (song_sim * 0.60) + (time_score * 0.25) + (artist_sim * 0.15)
    
    # Default to Non-Match (1)
    match_result = 1 
    
    # 4. Apply Threshold
    if total_score > 0.85:
        match_result = 0 # Match found
    
    results.append(match_result)

# Populate Result column
test['Result'] = results

# Save to CSV
test.to_csv('test_result_inverted_fast.csv', index=False)
print("Processing complete. Results saved to test_result_inverted_fast.csv")