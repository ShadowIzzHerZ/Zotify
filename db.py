"""Postgres storage: users, listening history, searches, feedback."""
import os

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

load_dotenv()

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id SERIAL PRIMARY KEY, name TEXT UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS listens(
  id SERIAL PRIMARY KEY, user_id INT NOT NULL REFERENCES users(id),
  item_id INT NOT NULL, source TEXT NOT NULL, ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(user_id, item_id));
CREATE TABLE IF NOT EXISTS searches(
  id SERIAL PRIMARY KEY, user_id INT NOT NULL REFERENCES users(id),
  query TEXT NOT NULL, rewritten TEXT, lam DOUBLE PRECISION NOT NULL, ts TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS feedback(
  id SERIAL PRIMARY KEY, user_id INT NOT NULL REFERENCES users(id),
  item_id INT NOT NULL, action TEXT NOT NULL CHECK(action IN ('play','save','skip')),
  query TEXT, lam DOUBLE PRECISION, ts TIMESTAMPTZ NOT NULL DEFAULT now());
"""


def conn():
    return psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row)


def init(seed_items):
    """Create tables and the demo user; seed its history once."""
    with conn() as c:
        c.execute(SCHEMA)
        c.execute("INSERT INTO users(name) VALUES('demo') ON CONFLICT DO NOTHING")
        uid = c.execute("SELECT id FROM users WHERE name='demo'").fetchone()["id"]
        if not c.execute("SELECT 1 FROM listens WHERE user_id=%s LIMIT 1", (uid,)).fetchone():
            with c.cursor() as cur:
                cur.executemany("INSERT INTO listens(user_id,item_id,source) VALUES(%s,%s,'seed') ON CONFLICT DO NOTHING",
                                [(uid, int(i)) for i in seed_items])
    return uid


def profile_items(uid):
    with conn() as c:
        return [r["item_id"] for r in c.execute("SELECT item_id FROM listens WHERE user_id=%s", (uid,))]


def log_search(uid, query, rewritten, lam):
    with conn() as c:
        c.execute("INSERT INTO searches(user_id,query,rewritten,lam) VALUES(%s,%s,%s,%s)", (uid, query, rewritten, lam))


def log_feedback(uid, item_id, action, query, lam):
    with conn() as c:
        c.execute("INSERT INTO feedback(user_id,item_id,action,query,lam) VALUES(%s,%s,%s,%s,%s)",
                  (uid, item_id, action, query, lam))
        if action in ("play", "save"):  # liked tracks become part of the profile
            c.execute("INSERT INTO listens(user_id,item_id,source) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING",
                      (uid, item_id, action))


def recent(uid, n=15):
    with conn() as c:
        return [dict(r, ts=r["ts"].timestamp()) for r in c.execute(
            "SELECT item_id, action, query, ts FROM feedback WHERE user_id=%s ORDER BY id DESC LIMIT %s", (uid, n))]


def stats(uid):
    with conn() as c:
        out = {"profile_size": c.execute("SELECT COUNT(*) n FROM listens WHERE user_id=%s", (uid,)).fetchone()["n"]}
        for a in ("play", "save", "skip"):
            out[a] = c.execute("SELECT COUNT(*) n FROM feedback WHERE user_id=%s AND action=%s", (uid, a)).fetchone()["n"]
        return out
