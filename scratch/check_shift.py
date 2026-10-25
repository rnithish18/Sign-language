import pandas as pd
d = pd.read_csv('data/processed/asl_landmarks.csv', low_memory=False)
feats = [f'f{i}' for i in range(128)]
w = d[d.source == 'webcam']
bad = w[pd.to_numeric(w['f127'], errors='coerce').isna()]
good = w[~w.index.isin(bad.index)]
print('webcam rows with empty/non-numeric f127:', len(bad), 'of', len(w))
print('labels affected:', sorted(bad.label.unique()))
print('labels clean   :', sorted(good.label.unique()))
print('session_id sample (bad rows):', bad['session_id'].head(3).tolist())
print('session_id sample (good rows):', good['session_id'].head(3).tolist())
print('cols with non-numeric in bad rows:',
      [c for c in feats if pd.to_numeric(bad[c], errors='coerce').isna().all()])