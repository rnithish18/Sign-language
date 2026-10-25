p = r"src\data_collection.py"
s = open(p, encoding="utf-8", newline="").read()

old = 'row = landmarks_to_csv_row(feat_vec, current_label, source="webcam")'
new = 'row = landmarks_to_csv_row(feat_vec, current_label, source="webcam", session_id="webcam_session3")'

assert s.count(old) == 1, "expected exactly one match, found %d" % s.count(old)
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new))
print("patched OK")