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
    """Lowercase, remove bracketed tags like [Explicit], keep alnum/spaces, collapse whitespace."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # remove bracketed content: [explicit], (live), etc.
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text)
    # keep letters/numbers/spaces only
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def sim(a, b):
    """Character-level similarity (edit-distance style) using SequenceMatcher ratio."""
    a = clean_text(a)
    b = clean_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

THRESHOLD = 0.6

def match_rule(row):
    song_sim = sim(row.get('Song_Name_A', ''), row.get('Song_Name_B', ''))
    artist_sim = sim(row.get('Artist_Name_A', ''), row.get('Artist_Name_B', ''))
    score = max(song_sim, artist_sim)
    # Result = 0 means match, Result = 1 means not match
    return 0 if score > THRESHOLD else 1

# Apply rule
merged['Result'] = merged.apply(match_rule, axis=1)

# Export just the required columns (keep any existing columns in test)
out = test.copy()
out['Result'] = merged['Result'].values

out.to_csv('entity_matching_results.csv', index=False)
print("Saved: entity_matching_results.csv")
