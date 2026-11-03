"""Local LLM (Ollama) for query rewriting and one-line explanations."""
import json
import os
import urllib.request

import pandas as pd

MODEL = "qwen2.5:3b"
REMOTE_MODEL = os.environ.get("REMOTE_MODEL", "qwen/qwen-2.5-7b-instruct")
GENRES = ", ".join(sorted(pd.read_parquet("data/catalog.parquet").track_genre.unique()))

REWRITE = f"""Turn a music request into one line describing the ideal track, in exactly this style:
Genre: <one genre>. <energy>, <mood>, <sound>, <tempo>, <vocals>.
energy: very low energy | low energy | medium energy | high energy | very high energy
mood: dark and sad | melancholic | neutral mood | upbeat | very happy
sound: electronic | mostly produced | mostly acoustic | acoustic
tempo: slow | mid tempo | fast
vocals: instrumental | vocal
Genre must be one of: {GENRES}
Output only that line.
Example request: sad piano for a rainy night
Example output: Genre: piano. low energy, melancholic, acoustic, slow, instrumental."""


def _post(url, body, headers):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def chat(system, user):
    """OpenRouter when OPENROUTER_API_KEY is set (deployed), else local Ollama."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        d = _post("https://openrouter.ai/api/v1/chat/completions",
                  {"model": REMOTE_MODEL, "temperature": 0, "messages": msgs}, {"Authorization": f"Bearer {key}"})
        return d["choices"][0]["message"]["content"].strip()
    d = _post("http://localhost:11434/api/chat",
              {"model": MODEL, "stream": False, "options": {"temperature": 0}, "messages": msgs}, {})
    return d["message"]["content"].strip()


def rewrite(query):
    return chat(REWRITE, query)


def explain(query, row, label="familiar"):
    sys = ("Write ONE short sentence saying why this track fits the request. "
           "Use only the facts given and quote the description's words exactly. Do not invent anything about the artist or lyrics. "
           "If the description clashes with the request, say it is a looser match and why.")
    facts = (f"Request: {query}\nTrack: {row.track_name} by {row.artists}\nGenre: {row.track_genre}\n"
             f"Description: {row.text}\nResult type: {label}")
    return chat(sys, facts)
