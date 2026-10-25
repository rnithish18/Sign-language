p = r"app.py"
s = open(p, encoding="utf-8", newline="").read()

old = 'media_stream_constraints={"video": True, "audio": False},'
new = ('media_stream_constraints={"video": {"width": {"ideal": 640}, '
       '"height": {"ideal": 480}, "frameRate": {"ideal": 15}}, "audio": False},')

assert s.count(old) == 2, "expected 2 matches, found %d" % s.count(old)
open(p + ".bak", "w", encoding="utf-8", newline="").write(s)
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new))
print("patched 2 places; backup at app.py.bak")