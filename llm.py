"""Local LLM (Ollama) for query rewriting and one-line explanations."""
import json
import urllib.request

import pandas as pd

MODEL = "qwen2.5:3b"
GENRES = ", ".join(sorted(pd.read_parquet("data/catalog.parquet").track_genre.unique()))

REWRITE = f"""Turn a music request into one line describing the ideal track, in exactly this style:
Genre: <one genre>. <energy>, <mood>, <sound>, <tempo>, <vocals>.
energy: very low energy | low energy | medium energy | high energy | very high energy
mood: dark and sad | melancholic | neutral mood | upbeat | very happy
sound: electronic | mostly produced | mostly acoustic | acoustic
tempo: slow | mid tempo | fast
vocals: instrumental | vocal
Genre must be one of: {GENRES}
Output only that line."""


def chat(system, user):
    body = json.dumps({"model": MODEL, "stream": False, "options": {"temperature": 0},
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["message"]["content"].strip()


def rewrite(query):
    return chat(REWRITE, query)


def explain(query, row, label="familiar"):
    sys = ("Write ONE short sentence saying why this track fits the request. "
           "Use only the facts given. Do not invent anything about the artist or lyrics.")
    facts = (f"Request: {query}\nTrack: {row.track_name} by {row.artists}\nGenre: {row.track_genre}\n"
             f"Description: {row.text}\nResult type: {label}")
    return chat(sys, facts)
