"""FastAPI backend: search, explain, feedback, history. Serves the static frontend."""
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import db
import llm
import recommend as rec

app = FastAPI(title="Zotify")
UID = db.init(rec.profile)


class Search(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    lam: float = Field(0.0, ge=-1, le=1)
    k: int = Field(10, ge=1, le=30)
    use_llm: bool = True


class Explain(BaseModel):
    item_id: int
    query: str
    familiarity: float = 0.0


class Feedback(BaseModel):
    item_id: int
    action: str = Field(pattern="^(play|save|skip)$")
    query: str = ""
    lam: float = 0.0


def card(i, fam):
    r = rec.cat.iloc[i]
    return {"id": int(i), "title": r.track_name, "artists": r.artists.replace(";", ", "), "album": r.album_name,
            "genre": r.track_genre, "popularity": int(r.popularity), "familiarity": float(fam),
            "sid": ids_str(i), "energy": float(r.energy), "valence": float(r.valence),
            "acousticness": float(r.acousticness)}


def ids_str(i):
    return "-".join(str(int(c)) for c in rec.ids[i, :3])


@app.post("/api/search")
def search(s: Search):
    rewritten = None
    q = rec.embed_query(s.query)
    if s.use_llm:
        try:
            rewritten = llm.rewrite(s.query)
            q = q + rec.embed_query(rewritten)
            q = q / np.linalg.norm(q)
        except Exception:
            rewritten = None  # Ollama down: fall back to the raw query
    items, fam = rec.recommend(s.query, s.k, s.lam, q, db.profile_items(UID))
    db.log_search(UID, s.query, rewritten, s.lam)
    return {"rewritten": rewritten, "results": [card(i, f) for i, f in zip(items, fam)]}


@app.post("/api/explain")
def explain(e: Explain):
    row = rec.cat.iloc[e.item_id]
    label = "familiar (close to the listener's usual taste)" if e.familiarity >= 0.5 else "non-habitual (outside the listener's usual taste)"
    try:
        return {"text": llm.explain(e.query, row, label)}
    except Exception:
        return {"text": f"Matches your request on genre ({row.track_genre}) and sound: {row.text.split('. ', 2)[-1]}"}


@app.post("/api/feedback")
def feedback(f: Feedback):
    db.log_feedback(UID, f.item_id, f.action, f.query, f.lam)
    return db.stats(UID)


@app.get("/api/history")
def history():
    out = []
    for r in db.recent(UID):
        row = rec.cat.iloc[r["item_id"]]
        out.append({**r, "title": row.track_name, "artists": row.artists.replace(";", ", ")})
    return {"stats": db.stats(UID), "recent": out}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
