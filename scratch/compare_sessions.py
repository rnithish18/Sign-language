import pandas as pd
from sklearn.ensemble import RandomForestClassifier

old = pd.read_csv('data/backup/asl_landmarks_before_rerecord.csv', low_memory=False)
new = pd.read_csv('data/processed/asl_landmarks.csv', low_memory=False)
print('old cols:', list(old.columns[:4]), len(old.columns))
print('new cols:', list(new.columns[:4]), len(new.columns))

L = 'A B D E F G H I K L N Q R S T U V W X Z SPACE CLEAR'.split()
feats = [c for c in new.columns if c[0] == 'f' and c[1:].isdigit()]
o = old[(old.source == 'webcam') & old.label.isin(L)]
n = new[(new.source == 'webcam') & new.label.isin(L)]
X = lambda d: d[feats].apply(pd.to_numeric, errors='coerce').fillna(0)
print('rows old/new:', len(o), len(n))
print('non-numeric cells old/new:',
      int(o[feats].apply(pd.to_numeric, errors='coerce').isna().sum().sum()),
      int(n[feats].apply(pd.to_numeric, errors='coerce').isna().sum().sum()))

def fit(a): return RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(X(a), a.label)
print('train OLD -> test NEW:', (fit(o).predict(X(n)) == n.label).mean())
print('train NEW -> test OLD:', (fit(n).predict(X(o)) == o.label).mean())