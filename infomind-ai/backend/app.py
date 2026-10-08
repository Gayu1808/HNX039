"""InfoMind AI - FastAPI backend. Local TF-IDF retrieval + rule/NLP analysis; optional LLM via env."""
import os, re, io, json, sqlite3, hashlib, secrets, time, urllib.request
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DB = os.environ.get("INFOMIND_DB", str(ROOT / "infomind.db"))
MAX_MB, EXTS = 10, {"pdf", "docx", "txt", "csv", "png", "jpg", "jpeg"}
NOT_FOUND = "I couldn't find this information in the uploaded documents."
app = FastAPI(title="InfoMind AI")
con = sqlite3.connect(DB, check_same_thread=False); con.row_factory = sqlite3.Row

def q(sql, a=()): return [dict(r) for r in con.execute(sql, a).fetchall()]
def x(sql, a=()):
    c = con.execute(sql, a); con.commit(); return c.lastrowid

con.executescript("""
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE, pw TEXT, salt TEXT);
CREATE TABLE IF NOT EXISTS docs(id INTEGER PRIMARY KEY, name TEXT, type TEXT, pages INT, uploaded REAL, status TEXT, error TEXT, summary TEXT, meta TEXT, secs REAL);
CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, doc_id INT REFERENCES docs(id) ON DELETE CASCADE, page INT, text TEXT);
CREATE TABLE IF NOT EXISTS findings(id INTEGER PRIMARY KEY, doc_id INT, severity TEXT, kind TEXT, title TEXT, detail TEXT, source TEXT, action TEXT);
CREATE TABLE IF NOT EXISTS actions(id INTEGER PRIMARY KEY, finding_id INT, title TEXT, priority TEXT DEFAULT 'medium', assignee TEXT DEFAULT '', note TEXT DEFAULT '', status TEXT DEFAULT 'open', created REAL);
CREATE TABLE IF NOT EXISTS searches(id INTEGER PRIMARY KEY, query TEXT, kind TEXT, ts REAL);
""")
def hpw(p, s): return hashlib.sha256((s + p).encode()).hexdigest()
if not q("SELECT 1 FROM users"):
    s = secrets.token_hex(8); x("INSERT INTO users(username,pw,salt) VALUES('admin',?,?)", (hpw(os.environ.get("ADMIN_PASSWORD", "admin123"), s), s))
TOKENS = {}
def auth(x_token: str = Header(default="")):
    if x_token not in TOKENS: raise HTTPException(401, "Please sign in again.")
    return TOKENS[x_token]

# ---------- extraction ----------
def extract(name, data):
    ext = name.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        from pypdf import PdfReader
        pages = [(p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages]
    elif ext == "docx":
        import docx
        pages = ["\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)]
    elif ext in ("txt", "csv"):
        pages = [data.decode("utf-8", "ignore")]
    else:
        try:
            import pytesseract; from PIL import Image
            pages = [pytesseract.image_to_string(Image.open(io.BytesIO(data)))]
        except Exception: raise ValueError("OCR failed: install Tesseract OCR to read scanned images.")
    if not "".join(pages).strip(): raise ValueError("No readable text found in this file.")
    return ext, pages

# ---------- analysis ----------
FIELD = re.compile(r"^\s*([A-Za-z][A-Za-z ]{1,30}?)\s*:\s*(.+?)\s*$")
AMT = re.compile(r"(?:₹|Rs\.?|INR|\$)\s?\d[\d,]*(?:\.\d+)?", re.I)
DATE = re.compile(r"\b\d{1,2}(?:st|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*(?:\s+\d{4})?|\b\d{4}-\d{2}-\d{2}\b", re.I)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"); PHONE = re.compile(r"\b(?:\+91[\s-]?)?\d{5}[\s-]?\d{5}\b")
SKIP = {"name", "employee", "applicant", "candidate", "phone", "email", "report", "title", "document", "attached documents", "remarks", "department", "project", "skill"}
REQUIRED = {"Name": r"^(name|applicant|candidate)\s*:", "Phone": PHONE.pattern, "Email": EMAIL.pattern,
            "Income Certificate": "income certificate", "Address Proof": "address proof", "ID Proof": "id proof"}
NEG = ("missing", "pending", "not provided", "not submitted", "not attached")
def num(s): return float(re.sub(r"[^\d.]", "", s) or 0)
def norm(v):
    v = re.sub(r"[^a-z0-9]", "", v.lower()); m = re.fullmatch(r"(?:rs|inr|usd)?(\d+)", v); return m.group(1) if m else v
SEV = {"critical": 0, "important": 1, "review": 2, "info": 3}

def analyze(did):
    d = q("SELECT * FROM docs WHERE id=?", (did,))[0]
    chunks = q("SELECT page,text FROM chunks WHERE doc_id=? ORDER BY id", (did,))
    full = "\n".join(c["text"] for c in chunks); lines = [l for l in full.split("\n") if l.strip()]
    fields = {}
    for l in lines:
        m = FIELD.match(l)
        if m: fields[m.group(1).strip().lower()] = m.group(2).strip()
    subj = next((fields[k] for k in ("name", "employee", "applicant", "candidate") if k in fields), "")
    amts = AMT.findall(full)
    meta = dict(fields=fields, subject=subj, amounts=amts, dates=sorted(set(DATE.findall(full))), emails=sorted(set(EMAIL.findall(full))),
                phones=sorted(set(PHONE.findall(full))), subject_name=subj or d["name"])
    summ = " ".join(lines[:3])[:240]
    x("UPDATE docs SET meta=?, summary=? WHERE id=?", (json.dumps(meta), summ, did))
    x("DELETE FROM findings WHERE doc_id=? AND kind!='conflict'", (did,))
    def f(sev, kind, title, detail, src, act=""): x("INSERT INTO findings(doc_id,severity,kind,title,detail,source,action) VALUES(?,?,?,?,?,?,?)", (did, sev, kind, title, detail, src, act))
    for k, v in fields.items():
        if "deadline" in k or k.startswith("due"):
            f("critical", "deadline", f"{k.title()}: {v}", f"{d['name']} sets a {k} of {v}.", f"{d['name']}, line “{k.title()}: {v}”", f"Track {k}: {v} ({d['name']})")
    if "application" in d["name"].lower() or "application" in full[:300].lower():
        who = subj or d["name"]; miss = []
        for item, pat in REQUIRED.items():
            hit = [l for l in lines if re.search(pat, l, re.I) and not any(n in l.lower() for n in NEG)]
            if not hit: miss.append(item)
        for item in miss:
            f("important", "missing", f"{who}: {item} not found", f"No evidence of “{item}” in {d['name']}; it may be missing.", d["name"], f"Request {item} from {who}")
    if amts:
        top = max(amts, key=num); f("info", "amount", f"Highest amount: {top}", f"{d['name']} mentions {len(amts)} amounts; largest is {top}.", d["name"])
    f("info", "info", f"{d['name']} processed", summ or "Document processed.", d["name"])
    cross()

def cross():
    x("DELETE FROM findings WHERE kind='conflict'"); seen = {}
    for d in q("SELECT id,name,meta FROM docs WHERE status='ready'"):
        m = json.loads(d["meta"] or "{}")
        for k, v in m.get("fields", {}).items():
            if k in SKIP: continue
            s = "" if ("deadline" in k) else m["subject"].lower()
            seen.setdefault((s, k), []).append((d["id"], d["name"], v, m["subject"]))
    for (s, k), L in seen.items():
        if len({norm(v) for _, _, v, _ in L}) > 1:
            who = f" for {L[0][3]}" if s else ""
            det = " vs ".join(f"{n} → {v}" for _, n, v, _ in L)
            x("INSERT INTO findings(doc_id,severity,kind,title,detail,source,action) VALUES(?,?,?,?,?,?,?)",
              (L[0][0], "review", "conflict", f"Possible conflict: {k.title()}{who}", det, "; ".join(n for _, n, _, _ in L), f"Verify {k}{who} with document owners"))

def ingest(name, data):
    t0 = time.time(); ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    did = x("INSERT INTO docs(name,type,pages,uploaded,status) VALUES(?,?,0,?,'processing')", (name, ext, time.time()))
    try:
        if ext not in EXTS: raise ValueError(f"Unsupported format .{ext}. Use PDF, DOCX, TXT, CSV or an image.")
        if len(data) > MAX_MB * 1048576: raise ValueError(f"File is larger than {MAX_MB} MB.")
        ext, pages = extract(name, data)
        for i, p in enumerate(pages, 1):
            for part in re.split(r"\n\s*\n", p):
                if part.strip(): x("INSERT INTO chunks(doc_id,page,text) VALUES(?,?,?)", (did, i, part.strip()))
        x("UPDATE docs SET pages=?, status='ready', secs=? WHERE id=?", (len(pages), 0, did)); analyze(did)
        x("UPDATE docs SET secs=? WHERE id=?", (round(time.time() - t0, 3), did))
    except Exception as e:
        x("UPDATE docs SET status='error', error=? WHERE id=?", (str(e), did))
    return did

# ---------- retrieval ----------
def sentences():
    out = []
    for c in q("SELECT c.page,c.text,d.name,d.id did FROM chunks c JOIN docs d ON d.id=c.doc_id WHERE d.status='ready'"):
        for s in re.split(r"\n|(?<=[.!?])\s+", c["text"]):
            if len(s.strip()) > 8: out.append(dict(doc=c["name"], did=c["did"], page=c["page"], text=s.strip(), chunk=c["text"]))
    return out
SYN = {"payment": "salary pay amount deadline", "pay": "salary payment", "incomplete": "missing pending", "performance": "rating project review",
       "complaints": "complaint customer issue", "employee": "applicant staff", "salary": "pay payment amount"}
def expand(s): return s + " " + " ".join(SYN.get(w, "") for w in re.findall(r"\w+", s.lower()))
def retrieve(query, k=5, min_score=0.1):
    S = sentences()
    if not S: return []
    v = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2)).fit([s["text"] for s in S] + [query])
    sc = cosine_similarity(v.transform([expand(query)]), v.transform([s["text"] for s in S]))[0]
    rank = sorted(zip(sc, S), key=lambda t: -t[0]); res, seen = [], set()
    for s, t in rank:
        if s < min_score or len(res) >= k: break
        if t["text"] in seen: continue
        seen.add(t["text"]); res.append(dict(score=round(float(s), 2), doc=t["doc"], did=t["did"], page=t["page"], snippet=t["text"], section=t["chunk"][:300]))
    return res

class Login(BaseModel): username: str; password: str
class Ask(BaseModel): q: str
class Cmp(BaseModel): ids: list[int]
class ActIn(BaseModel): finding_id: int | None = None; title: str = ""
class ActUp(BaseModel): priority: str | None = None; assignee: str | None = None; note: str | None = None; status: str | None = None

@app.post("/api/login")
def login(b: Login):
    u = q("SELECT * FROM users WHERE username=?", (b.username,))
    if not u or u[0]["pw"] != hpw(b.password, u[0]["salt"]): raise HTTPException(401, "Wrong username or password.")
    t = secrets.token_hex(16); TOKENS[t] = b.username; return {"token": t}

@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...), u=Depends(auth)):
    ids = [ingest(f.filename, await f.read()) for f in files]; return {"ids": ids}
@app.post("/api/demo")
def demo(u=Depends(auth)):
    from demo_data import DOCS
    have = {d["name"] for d in q("SELECT name FROM docs")}
    for n, t in DOCS.items():
        if n not in have: ingest(n, t.encode())
    return {"ok": True}
@app.get("/api/docs")
def docs(u=Depends(auth)): return q("SELECT id,name,type,pages,uploaded,status,error,summary,secs FROM docs ORDER BY id DESC")
@app.get("/api/docs/{i}")
def doc(i: int, u=Depends(auth)):
    d = q("SELECT * FROM docs WHERE id=?", (i,))
    if not d: raise HTTPException(404, "Document not found.")
    d = d[0]; d["meta"] = json.loads(d["meta"] or "{}")
    d["findings"] = q("SELECT * FROM findings WHERE doc_id=?", (i,)); d["text"] = "\n\n".join(c["text"] for c in q("SELECT text FROM chunks WHERE doc_id=?", (i,)))
    return d
@app.delete("/api/docs/{i}")
def rm(i: int, u=Depends(auth)):
    x("DELETE FROM chunks WHERE doc_id=?", (i,)); x("DELETE FROM findings WHERE doc_id=?", (i,)); x("DELETE FROM docs WHERE id=?", (i,)); cross(); return {"ok": True}

@app.get("/api/search")
def search(q_: str = "", u=Depends(auth)):
    qq = q_.strip()
    if not qq: raise HTTPException(400, "Type something to search.")
    x("INSERT INTO searches(query,kind,ts) VALUES(?,?,?)", (qq, "search", time.time())); return retrieve(qq, 8, 0.05)

def llm(question, ctx):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key: return None
    body = json.dumps({"model": os.environ.get("LLM_MODEL", "claude-sonnet-4-6"), "max_tokens": 600,
        "system": f"Answer ONLY from the provided sources. If they don't contain the answer reply exactly: {NOT_FOUND}",
        "messages": [{"role": "user", "content": f"Sources:\n{ctx}\n\nQuestion: {question}"}]}).encode()
    try:
        r = urllib.request.Request("https://api.anthropic.com/v1/messages", body, {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        return json.load(urllib.request.urlopen(r, timeout=30))["content"][0]["text"]
    except Exception: return None

@app.post("/api/ask")
def ask(b: Ask, u=Depends(auth)):
    ql = b.q.lower(); x("INSERT INTO searches(query,kind,ts) VALUES(?,?,?)", (b.q, "ask", time.time()))
    def fk(kind): return q("SELECT f.*, d.name dname FROM findings f JOIN docs d ON d.id=f.doc_id WHERE kind=?", (kind,))
    intent = next((k for w, k in (("incomplete", "missing"), ("missing", "missing"), ("conflict", "conflict"), ("inconsisten", "conflict"), ("deadline", "deadline")) if w in ql), None)
    if intent:
        F = fk(intent)
        if not F: return {"answer": f"No {intent} items were detected in the uploaded documents.", "sources": []}
        head = {"missing": "possible missing items", "conflict": "possible conflicts", "deadline": "deadlines"}[intent]
        return {"answer": f"{len(F)} {head} found:\n" + "\n".join(f"• {f['title']} — {f['detail']}" for f in F),
                "sources": [dict(doc=f["dname"], page=1, snippet=f["detail"], score=1.0) for f in F]}
    if re.search(r"(highest|largest|maximum|biggest).*(amount|salary|payment)", ql):
        best = None
        for d in q("SELECT name,meta FROM docs WHERE status='ready'"):
            for a in json.loads(d["meta"] or "{}").get("amounts", []):
                if not best or num(a) > num(best[1]): best = (d["name"], a)
        if best: return {"answer": f"{best[0]} mentions the highest amount: {best[1]}.", "sources": [dict(doc=best[0], page=1, snippet=best[1], score=1.0)]}
    R = retrieve(b.q, 4, 0.12)
    if not R: return {"answer": NOT_FOUND, "sources": []}
    a = llm(b.q, "\n".join(f"[{r['doc']} p{r['page']}] {r['section']}" for r in R)) or "Based on the documents:\n" + "\n".join(f"• {r['snippet']} ({r['doc']}, p{r['page']})" for r in R[:3])
    return {"answer": a, "sources": R}

@app.post("/api/compare")
def compare(b: Cmp, u=Depends(auth)):
    if len(b.ids) < 2: raise HTTPException(400, "Select at least two documents.")
    D = [(d["name"], json.loads(d["meta"] or "{}")) for d in (q("SELECT name,meta FROM docs WHERE id=?", (i,))[0] for i in b.ids)]
    keys = {k for _, m in D for k in m["fields"]}; out = dict(added=[], removed=[], changed=[], common=[], conflicts=[], important=[])
    base_n, base = D[0]
    for k in sorted(keys):
        vals = [(n, m["fields"].get(k)) for n, m in D]; present = [(n, v) for n, v in vals if v]
        if len(present) == len(D) and len({norm(v) for _, v in present}) == 1: out["common"].append(dict(field=k, value=present[0][1]))
        elif len(present) == len(D):
            row = dict(field=k, values=[dict(doc=n, value=v) for n, v in present]); out["changed"].append(row)
            if k not in SKIP: out["conflicts"].append(row); out["important"].append(row)
        else:
            (out["added"] if not base["fields"].get(k) else out["removed"]).append(dict(field=k, values=[dict(doc=n, value=v) for n, v in present]))
    return out

@app.get("/api/findings")
def findings(severity: str = "", u=Depends(auth)):
    r = q("SELECT f.*, d.name dname FROM findings f JOIN docs d ON d.id=f.doc_id" + (" WHERE severity=?" if severity else ""), (severity,) if severity else ())
    return sorted(r, key=lambda f: SEV[f["severity"]])
@app.get("/api/actions")
def actions(u=Depends(auth)): return q("SELECT * FROM actions ORDER BY status='done', id DESC")
@app.post("/api/actions")
def mk_action(b: ActIn, u=Depends(auth)):
    t, p = b.title, "medium"
    if b.finding_id:
        f = q("SELECT * FROM findings WHERE id=?", (b.finding_id,))
        if not f: raise HTTPException(404, "Finding not found.")
        t = f[0]["action"] or f"Review: {f[0]['title']}"; p = {"critical": "high", "important": "high"}.get(f[0]["severity"], "medium")
    if not t: raise HTTPException(400, "Action needs a title.")
    return {"id": x("INSERT INTO actions(finding_id,title,priority,created) VALUES(?,?,?,?)", (b.finding_id, t, p, time.time()))}
@app.patch("/api/actions/{i}")
def up_action(i: int, b: ActUp, u=Depends(auth)):
    for k, v in b.model_dump(exclude_none=True).items(): x(f"UPDATE actions SET {k}=? WHERE id=?", (v, i))
    return {"ok": True}

@app.get("/api/dashboard")
def dash(u=Depends(auth)):
    sev = {r["severity"]: r["n"] for r in q("SELECT severity, COUNT(*) n FROM findings WHERE kind!='info' GROUP BY severity")}
    kinds = {r["kind"]: r["n"] for r in q("SELECT kind, COUNT(*) n FROM findings GROUP BY kind")}
    return dict(total=q("SELECT COUNT(*) n FROM docs")[0]["n"], processed=q("SELECT COUNT(*) n FROM docs WHERE status='ready'")[0]["n"],
        important=sum(v for k, v in sev.items() if k in ("critical", "important")), conflicts=kinds.get("conflict", 0), missing=kinds.get("missing", 0),
        severity=sev, types={r["type"]: r["n"] for r in q("SELECT type, COUNT(*) n FROM docs GROUP BY type")},
        avg_secs=(q("SELECT AVG(secs) a FROM docs WHERE status='ready'")[0]["a"] or 0), searches=q("SELECT COUNT(*) n FROM searches")[0]["n"],
        recent_docs=q("SELECT id,name,status,uploaded FROM docs ORDER BY id DESC LIMIT 5"), recent_searches=q("SELECT query,kind FROM searches ORDER BY id DESC LIMIT 5"),
        open_actions=q("SELECT COUNT(*) n FROM actions WHERE status!='done'")[0]["n"])

@app.get("/", response_class=HTMLResponse)
def index(): return (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
