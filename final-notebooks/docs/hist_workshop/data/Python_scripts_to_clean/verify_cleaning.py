import pandas as pd
import re
from difflib import SequenceMatcher

# -----------------------
ACT_FILENAME = 'chinese_regulation_act_1884.txt'
SIM_THRESHOLD = 0.60  # match cleaning script
MIN_SENT_LEN = 30
CLEANED_CSV = 'metadata_cleaned.csv'
# -----------------------

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', str(text))
    return text.strip()

def split_sentences(text: str) -> list:
    sents = re.split(r'[.!?]+', text)
    return [clean_text(s) for s in sents if len(clean_text(s)) >= MIN_SENT_LEN]

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def main():
    df = pd.read_csv(CLEANED_CSV)
    act_row = df[df['filename'] == ACT_FILENAME]
    if act_row.empty:
        raise ValueError(f"Could not find row with filename={ACT_FILENAME}")
    act_text = str(act_row.iloc[0]['text'])
    act_sents = split_sentences(act_text)

    n_matches = 0
    for idx, row in df.iterrows():
        if row.get('author','') == 'Crease':
            sents = split_sentences(str(row['text']))
            for c_sent in sents:
                if any(similarity(c_sent, a_sent) >= SIM_THRESHOLD for a_sent in act_sents):
                    print(f"[WARNING] Possible missed quote in {row['filename']}:\n{c_sent[:120]}\n")
                    n_matches += 1
    if n_matches == 0:
        print("No Act quotes found remaining in any Crease file (above threshold).")
    else:
        print(f"{n_matches} possible missed quoted sentences remain.")

if __name__ == "__main__":
    main()
