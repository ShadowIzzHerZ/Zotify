"""SQLite storage: users, listening history, searches, feedback."""
import sqlite3
import time

PATH = "data/zotify.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS listens(
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
  item_id INTEGER NOT NULL, source TEXT NOT NULL, ts REAL NOT NULL,
  UNIQUE(user_id, item_id));
CREATE TABLE IF NOT EXISTS searches(
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
  query TEXT NOT NULL, rewritten TEXT, lam REAL NOT NULL, ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS feedback(
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
  item_id INTEGER NOT NULL, action TEXT NOT NULL CHECK(action IN ('play','save','skip')),
  query TEXT, lam REAL, ts REAL NOT NULL);
"""


def conn():
    c = sqlite3.connect(PATH)
    c.row_factory = sqlite3.Row
    return c


def init(seed_items):
    """Create tables and the demo user; seed its history once."""
    with conn() as c:
        c.executescript(SCHEMA)
        c.execute("INSERT OR IGNORE INTO users(name) VALUES('demo')")
        uid = c.execute("SELECT id FROM users WHERE name='demo'").fetchone()["id"]
        if not c.execute("SELECT 1 FROM listens WHERE user_id=?", (uid,)).fetchone():
            c.executemany("INSERT OR IGNORE INTO listens(user_id,item_id,source,ts) VALUES(?,?,'seed',?)",
                          [(uid, int(i), time.time()) for i in seed_items])
    return uid


def profile_items(uid):
    with conn() as c:
        return [r["item_id"] for r in c.execute("SELECT item_id FROM listens WHERE user_id=?", (uid,))]


def log_search(uid, query, rewritten, lam):
    with conn() as c:
        c.execute("INSERT INTO searches(user_id,query,rewritten,lam,ts) VALUES(?,?,?,?,?)",
                  (uid, query, rewritten, lam, time.time()))


def log_feedback(uid, item_id, action, query, lam):
    with conn() as c:
        c.execute("INSERT INTO feedback(user_id,item_id,action,query,lam,ts) VALUES(?,?,?,?,?,?)",
                  (uid, item_id, action, query, lam, time.time()))
        if action in ("play", "save"):  # liked tracks become part of the profile
            c.execute("INSERT OR IGNORE INTO listens(user_id,item_id,source,ts) VALUES(?,?,?,?)",
                      (uid, item_id, action, time.time()))


def recent(uid, n=15):
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT item_id, action, query, ts FROM feedback WHERE user_id=? ORDER BY id DESC LIMIT ?", (uid, n))]


def stats(uid):
    with conn() as c:
        return {"profile_size": c.execute("SELECT COUNT(*) n FROM listens WHERE user_id=?", (uid,)).fetchone()["n"],
                **{a: c.execute("SELECT COUNT(*) n FROM feedback WHERE user_id=? AND action=?", (uid, a)).fetchone()["n"]
                   for a in ("play", "save", "skip")}}
