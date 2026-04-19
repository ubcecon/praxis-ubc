"""Score corpus documents with domain-specific stance lexicon."""
import pandas as pd
import re
from collections import Counter

df = pd.read_csv('data/metadata_cleaned.csv')
lexicon = pd.read_csv('data/stance_lexicon.csv')

lex_dict = {}
for _, row in lexicon.iterrows():
    lex_dict[row['term'].strip().lower()] = row['category'].strip()

CATS = [
    'EXCLUSIONARY', 'RIGHTS_AFFIRMING', 'DEHUMANIZING',
    'LEGAL_PROCEDURAL', 'ECONOMIC', 'SANITARY_MEDICAL',
]


def score_document(text, lex):
    text_lower = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text_lower)
    total = len(tokens)
    if total == 0:
        return {}
    counts = Counter()
    for t in tokens:
        if t in lex:
            counts[lex[t]] += 1
    for i in range(len(tokens) - 1):
        bigram = tokens[i] + ' ' + tokens[i + 1]
        if bigram in lex:
            counts[lex[bigram]] += 1
    result = {c: counts.get(c, 0) / total * 1000 for c in CATS}
    result['total_tokens'] = total
    result['lexicon_hits'] = sum(counts.values())
    return result


df['group'] = 'Other'
df.loc[df['author'] == 'Crease', 'group'] = 'Crease'
df.loc[df['author'] == 'Begbie', 'group'] = 'Begbie'
df.loc[df['type'] == 'act', 'group'] = 'Act'

scores = df['text'].apply(lambda t: score_document(t, lex_dict))
scores_df = pd.DataFrame(scores.tolist())
scores_df['group'] = df['group'].values
scores_df['filename'] = df['filename'].values

print('=== Per-Document Lexicon Scores (per 1,000 tokens) ===\n')
for _, row in scores_df.iterrows():
    print(f"{row['filename']} ({row['group']})")
    print(f"  Tokens: {int(row['total_tokens'])}, Hits: {int(row['lexicon_hits'])}")
    for c in CATS:
        print(f'  {c}: {row[c]:.2f}')
    print()

print('\n=== Group Means (per 1,000 tokens) ===\n')
means = scores_df.groupby('group')[CATS].mean()
print(means.round(2).to_string())

scores_df.to_csv('data/lexicon_scores.csv', index=False)
print('\nSaved to data/lexicon_scores.csv')
