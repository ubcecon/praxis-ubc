import pandas as pd
import re
from difflib import SequenceMatcher

# -----------------------
ACT_FILENAME = 'chinese_regulation_act_1884.txt'
SIM_THRESHOLD = 0.60  # remove anything 60% or more similar (adjust as needed)
MIN_SENT_LEN = 30     # only check/remove sentences this long or longer
INPUT_CSV = 'metadata.csv'
OUTPUT_CSV = 'metadata_cleaned.csv'
# -----------------------

def clean_text(text: str) -> str:
    # Remove extra whitespace, leave punctuation for matching
    text = re.sub(r'\s+', ' ', str(text))
    return text.strip()

def split_sentences(text: str) -> list:
    # Simple split on . ! ? followed by whitespace/newline (not perfect but OK for legal docs)
    sents = re.split(r'[.!?]+', text)
    return [clean_text(s) for s in sents if len(clean_text(s)) >= MIN_SENT_LEN]

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def remove_similar_sentences(text, act_sents):
    # Remove sentences from `text` that are too similar to any in act_sents
    sents = re.split(r'([.!?]+)', text) # Keep punctuation
    combined = [''.join(pair) for pair in zip(sents[0::2], sents[1::2])]  # sentence + delimiter
    if len(sents) % 2 == 1:
        combined.append(sents[-1])
    filtered = []
    for sent in combined:
        cleaned = clean_text(sent)
        if len(cleaned) >= MIN_SENT_LEN:
            if any(similarity(cleaned, act_s) >= SIM_THRESHOLD for act_s in act_sents):
                continue  # skip this, it's a quote
        filtered.append(sent)
    return ''.join(filtered)

def main():
    print(f"Reading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)
    # Get the Act text as basis for matching
    act_row = df[df['filename'] == ACT_FILENAME]
    if act_row.empty:
        raise ValueError(f"Could not find row with filename={ACT_FILENAME}")
    act_text = str(act_row.iloc[0]['text'])
    act_sents = split_sentences(act_text)
    print(f"Chinese Regulation Act parsed, {len(act_sents)} sentences")

    cleaned_texts = []
    removed_counts = []

    for idx, row in df.iterrows():
        # Only clean Crease's files
        if row.get('author','') == 'Crease':
            original_text = str(row['text'])
            sents = split_sentences(original_text)
            mask = []
            for s in sents:
                is_quote = any(similarity(s, act_s) >= SIM_THRESHOLD for act_s in act_sents)
                mask.append(is_quote)
            # Remove quoted sentences
            filtered_sents = [s for s,m in zip(sents,mask) if not m]
            # For proper output preserve original punctuation (see below)
            cleaned = remove_similar_sentences(original_text, act_sents)
            # Stats
            n_removed = sum(mask)
            chars_before = len(original_text)
            chars_after = len(cleaned)
            print(f"{row['filename']}: removed {n_removed} quoted Act sentences, "
                  f"{chars_before-chars_after} chars cut ({chars_before}->{chars_after})")
            cleaned_texts.append(cleaned)
            removed_counts.append(n_removed)
        else:
            cleaned_texts.append(row['text'])
            removed_counts.append(0)

    df['text'] = cleaned_texts
    df['act_quote_sentences_removed'] = removed_counts
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! Cleaned file written to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
