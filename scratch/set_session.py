import re, sys
p = r"src\data_collection.py"
new_id = sys.argv[1]
s = open(p, encoding="utf-8", newline="").read()
s2, n = re.subn(r'session_id="[^"]*"', 'session_id="%s"' % new_id, s)
assert n == 1, "expected exactly one session_id, found %d" % n
open(p, "w", encoding="utf-8", newline="").write(s2)
print("session tag set to", new_id)