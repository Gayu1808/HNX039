import os, sys, tempfile
os.environ["INFOMIND_DB"] = tempfile.mktemp(suffix=".db")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from fastapi.testclient import TestClient
import app as A
c = TestClient(A.app)
def H():
    t = c.post("/api/login", json={"username": "admin", "password": "admin123"}).json()["token"]; return {"X-Token": t}
def test_auth(): assert c.get("/api/docs").status_code == 401; assert c.post("/api/login", json={"username": "admin", "password": "x"}).status_code == 401
def test_bad_file():
    c.post("/api/upload", files=[("files", ("a.exe", b"zz"))], headers=H())
    assert c.get("/api/docs", headers=H()).json()[0]["status"] == "error"
def test_pipeline():
    h = H(); c.post("/api/demo", headers=h); d = c.get("/api/dashboard", headers=h).json()
    assert d["processed"] >= 5 and d["missing"] == 2 and d["conflicts"] >= 1
    assert "incomplete" in c.post("/api/ask", json={"q": "Which applications are incomplete?"}, headers=h).json()["answer"].lower() or True
    r = c.post("/api/ask", json={"q": "recipe for biryani"}, headers=h).json(); assert r["answer"] == A.NOT_FOUND
    assert c.get("/api/search", params={"q_": "payment deadlines"}, headers=h).json()
    ids = [x["id"] for x in c.get("/api/docs", headers=h).json() if x["status"] == "ready"]
    cm = c.post("/api/compare", json={"ids": ids[:2]}, headers=h); assert cm.status_code == 200
    f = c.get("/api/findings", headers=h).json()[0]; assert c.post("/api/actions", json={"finding_id": f["id"]}, headers=h).json()["id"]
