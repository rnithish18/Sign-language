import sys, joblib, pandas as pd
from collections import Counter
from sklearn.metrics import classification_report

sess = sys.argv[1]
d = pd.read_csv(r"data\processed\asl_landmarks.csv", low_memory=False)
d = d[(d.source == "webcam") & (d.session_id == sess)]
print("rows in", sess, ":", len(d))

feat = [c for c in d.columns if c.startswith("f")]
model = joblib.load(r"models\signbridge_asl_model.joblib")
labels = joblib.load(r"models\signbridge_asl_labels.joblib")

pred = model.predict(d[feat].values.astype(float))
if hasattr(labels, "inverse_transform"):
    pred = labels.inverse_transform(pred)
elif len(pred) and not isinstance(pred[0], str):
    pred = [labels[i] for i in pred]

y = d.label.values
acc = sum(a == b for a, b in zip(y, pred)) / len(y)
print("HELD-OUT ACCURACY: %.1f%%" % (acc * 100))
print(classification_report(y, pred, zero_division=0))
print("Top confusions (true -> predicted):")
for (a, b), n in Counter((a, b) for a, b in zip(y, pred) if a != b).most_common(10):
    print("  %s -> %s : %d" % (a, b, n))
