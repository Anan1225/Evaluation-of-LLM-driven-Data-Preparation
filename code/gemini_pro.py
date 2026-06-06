import pandas as pd
import re

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
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^]]*\]', '', text)
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
    
    # Default to Non-Match (1)
    match_result = 1 
    
    # 3. Apply Matching Logic
    # Artists and Song Names must match
    if artist_a == artist_b and artist_a != "":
        if song_a == song_b and song_a != "":
            # Time must be within 10 seconds
            if time_a is not None and time_b is not None:
                if abs(time_a - time_b) <= 10:
                    match_result = 0 # Match found
    
    results.append(match_result)

# Populate Result column
test['Result'] = results

# Save to CSV
test.to_csv('test_result_inverted.csv', index=False)