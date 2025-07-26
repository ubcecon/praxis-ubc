import pandas as pd
import re
from difflib import SequenceMatcher

# -----------------------
ACT_FILENAME = 'chinese_regulation_act_1884.txt'
SIM_HIGH = 0.80   # near-identical
SIM_MED  = 0.60   # partial / paraphrase
MIN_SENT_LEN = 30 # ignore trivial sentences
# -----------------------

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return re.sub(r'[^\w\s\.\,\;\:\!\?\-\"\']', ' ', text).strip()

def split_sentences(text: str) -> list:
    sents = re.split(r'[.!?]+', text)
    return [clean_text(s) for s in sents if len(clean_text(s)) >= MIN_SENT_LEN]

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def main():
    df = pd.read_csv('metadata_cleaned.csv') #change here between the meta_data files
    crease_cases = df[df['author'] == 'Crease']
    act_row      = df[df['filename'] == ACT_FILENAME]
    if act_row.empty or crease_cases.empty:
        raise ValueError("Could not locate required rows in metadata.csv.")

    act_text = act_row.iloc[0]['text']
    act_sents = split_sentences(act_text)
    all_matches = []
    for _, row in crease_cases.iterrows():
        case_name = row['filename']
        crease_sents = split_sentences(row['text'])
        for ci, c_sent in enumerate(crease_sents):
            for ai, a_sent in enumerate(act_sents):
                score = similarity(c_sent, a_sent)
                if score >= SIM_MED:
                    all_matches.append({
                        'crease_case': case_name,
                        'similarity': round(score, 3),
                        'crease_sentence': c_sent,
                        'act_sentence': a_sent
                    })
    all_matches.sort(key=lambda x: x['similarity'], reverse=True)
    print(f"\nFound {len(all_matches)} overlapping or similar Crease/Act sentences\n")
    for i, m in enumerate(all_matches, 1):
        print("="*60)
        print(f"Match #{i}  (Sim: {m['similarity']}) in case '{m['crease_case']}'")
        print(f"Crease: {m['crease_sentence']}")
        print("-"*40)
        print(f"Act:    {m['act_sentence']}")
        print("="*60, "\n")

if __name__ == "__main__":
    main()
