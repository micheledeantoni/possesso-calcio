import sys, zipfile, re, html
for p in sys.argv[1:]:
    with zipfile.ZipFile(p) as z:
        x = z.read("word/document.xml").decode("utf8")
        names = [n for n in z.namelist() if n.startswith("word/media")]
    paras = []
    for para in re.findall(r"<w:p[ >].*?</w:p>", x, re.S):
        t = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", para, re.S))
        if t.strip(): paras.append(html.unescape(t))
    print("=" * 20, p, "| immagini:", len(names))
    print("\n".join(paras))
