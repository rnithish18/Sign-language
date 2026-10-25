import pandas as pd
p = 'data/processed/asl_landmarks.csv'
d = pd.read_csv(p, low_memory=False)
feats = [f'f{i}' for i in range(128)]
bad = (d.source == 'webcam') & pd.to_numeric(d['f127'], errors='coerce').isna()
print('rows to fix:', int(bad.sum()))
cols = ['session_id'] + feats[:-1]          # session_id holds f0, f0 holds f1, ... f126 holds f127
vals = d.loc[bad, cols].apply(pd.to_numeric, errors='coerce').values
d.loc[bad, feats] = vals
d.loc[bad, 'session_id'] = 'webcam_session2'
d.to_csv(p, index=False)
chk = pd.read_csv(p, low_memory=False)
w = chk[chk.source == 'webcam']
print('non-numeric/empty f0..f127 cells left:', int(w[feats].apply(pd.to_numeric, errors='coerce').isna().sum().sum()))
print(w.session_id.value_counts())