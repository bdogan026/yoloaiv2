"""label_tool.py — klavyeyle hizli tip etiketleme araci — image'a GIRMEZ.

datasets/tip_unsorted'taki crop'lari tek tek gosterir; 1-7 tuslariyla dogru tip
klasorune (datasets/tip/<tip>/) ANINDA tasir. Finder'da surukle-birak'tan 5-10x
hizli. Saf stdlib (http.server) — ek bagimlilik yok, tarayicida calisir.

Tuslar:
  1 sedan  2 suv  3 hatchback  4 pickup  5 minibus  6 panelvan  7 kamyon
  0 / Bosluk = ATLA (emin degilsen)   Z = geri al (son etiketi bozar)

Kullanim:
  python training/label_tool.py
  # tarayici otomatik acilir: http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.schema import ARAC_TIPLERI  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png")
CLASSES = list(ARAC_TIPLERI)


class _State:
    def __init__(self, src: str, out: str, skip: str) -> None:
        self.src, self.out, self.skip = src, out, skip
        self.history: list[tuple[str, str]] = []  # (hedef_yol, dosya_adi)

    def pending(self) -> list[str]:
        return sorted(f for f in os.listdir(self.src)
                      if f.lower().endswith(IMG_EXTS))

    def counts(self) -> dict:
        d = {c: len(os.listdir(os.path.join(self.out, c)))
             for c in CLASSES if os.path.isdir(os.path.join(self.out, c))}
        return d

    def move(self, fn: str, cls: str) -> None:
        srcp = os.path.join(self.src, fn)
        if not os.path.exists(srcp):
            return
        dest_dir = self.skip if cls == "_skip" else os.path.join(self.out, cls)
        os.makedirs(dest_dir, exist_ok=True)
        dstp = os.path.join(dest_dir, fn)
        shutil.move(srcp, dstp)
        self.history.append((dstp, fn))

    def undo(self) -> None:
        if not self.history:
            return
        dstp, fn = self.history.pop()
        if os.path.exists(dstp):
            shutil.move(dstp, os.path.join(self.src, fn))


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>tip etiketleme</title><style>
 body{background:#111;color:#eee;font-family:sans-serif;text-align:center;margin:0}
 #img{max-height:62vh;max-width:92vw;margin-top:10px;background:#222;border:1px solid #333}
 .keys{display:flex;flex-wrap:wrap;gap:6px;justify-content:center;margin:10px}
 .k{background:#223;padding:8px 12px;border-radius:6px;font-size:15px}
 .k b{color:#5f5;font-size:18px;margin-right:6px}
 #bar{color:#9cf;margin:6px;font-size:15px}
 #cts{color:#999;font-size:13px;margin:4px}
</style></head><body>
<div id=bar></div>
<img id=img>
<div class=keys id=keys></div>
<div id=cts></div>
<script>
const CLS=%s;
const keys=document.getElementById('keys');
CLS.forEach((c,i)=>keys.insertAdjacentHTML('beforeend',
  `<span class=k><b>${i+1}</b>${c}</span>`));
keys.insertAdjacentHTML('beforeend','<span class=k><b>0</b>atla</span><span class=k><b>Z</b>geri</span>');
let cur=null;
async function load(){
 const r=await fetch('/next');const j=await r.json();
 document.getElementById('cts').textContent=Object.entries(j.counts).map(([k,v])=>k+':'+v).join('  ');
 if(!j.file){document.getElementById('bar').textContent='BITTI — kalan 0';document.getElementById('img').src='';cur=null;return;}
 cur=j.file;
 document.getElementById('bar').textContent='kalan: '+j.remaining+'   |   '+j.file;
 document.getElementById('img').src='/img?f='+encodeURIComponent(j.file)+'&t='+Date.now();
}
async function send(cls){if(!cur&&cls!=='_undo')return;
 if(cls==='_undo'){await fetch('/undo');}else{await fetch('/label?f='+encodeURIComponent(cur)+'&cls='+cls);}
 load();}
document.addEventListener('keydown',e=>{
 if(e.key>='1'&&e.key<='7'){send(CLS[+e.key-1]);}
 else if(e.key==='0'||e.key===' '){e.preventDefault();send('_skip');}
 else if(e.key.toLowerCase()==='z'){send('_undo');}
});
load();
</script></body></html>"""


def make_handler(st: _State):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # sessiz
            pass

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/":
                self._send(200, "text/html; charset=utf-8",
                           (PAGE % json.dumps(CLASSES)).encode())
            elif u.path == "/next":
                p = st.pending()
                body = json.dumps({"file": p[0] if p else None,
                                   "remaining": len(p),
                                   "counts": st.counts()}).encode()
                self._send(200, "application/json", body)
            elif u.path == "/img":
                fn = (q.get("f") or [""])[0]
                fp = os.path.join(st.src, fn)
                if os.path.exists(fp):
                    with open(fp, "rb") as f:
                        self._send(200, "image/jpeg", f.read())
                else:
                    self._send(404, "text/plain", b"yok")
            elif u.path == "/label":
                st.move((q.get("f") or [""])[0], (q.get("cls") or [""])[0])
                self._send(200, "application/json", b'{"ok":true}')
            elif u.path == "/undo":
                st.undo()
                self._send(200, "application/json", b'{"ok":true}')
            else:
                self._send(404, "text/plain", b"yok")
    return H


def main() -> int:
    ap = argparse.ArgumentParser(description="klavyeyle hizli tip etiketleme")
    ap.add_argument("--src", default="datasets/tip_unsorted")
    ap.add_argument("--out", default="datasets/tip")
    ap.add_argument("--skip", default="datasets/tip_skip")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    if not os.path.isdir(args.src):
        print(f"kaynak yok: {args.src} (once sample_crops_for_labeling.py calistir)")
        return 1
    for c in CLASSES:
        os.makedirs(os.path.join(args.out, c), exist_ok=True)
    st = _State(args.src, args.out, args.skip)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(st))
    url = f"http://127.0.0.1:{args.port}"
    print(f"Etiketleme araci: {url}   (Ctrl+C ile bitir)")
    print(f"kalan: {len(st.pending())} crop")
    if not args.no_open:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbitti. sinif sayilari:", st.counts())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
