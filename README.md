# Zotify

Language-steerable music discovery. Describe what you want to hear ("something calm for a 3 a.m. study session") and get tracks from a catalog, each with a one-line explanation. A slider moves results between **familiar** (close to your listening history) and **non-habitual** (outside your routine).

**Live:** https://zotify.antideploy.app

Built for hackathon problem statement #8 (Spotify: LLM + semantic IDs + preference tuning).

## How it works

1. **Catalog**: 30,000 tracks sampled from the public Spotify tracks dataset on Hugging Face. Each track gets a short text built from its metadata and audio features ("high energy, upbeat, electronic, fast, vocal").
2. **Embeddings**: the text is embedded with `all-MiniLM-L6-v2`.
3. **Semantic IDs**: residual k-means over the embeddings gives each track a 3-level code (256 codes per level) plus a collision code. All IDs live in a trie.
4. **Search**: a local or hosted LLM rewrites the query into a catalog-style description. A beam search walks the trie, so only real catalog tracks can come out, then candidates are reranked by exact cosine similarity.
5. **Familiar / non-habitual switch**: a slider value `lam` in [-1, 1] raises or lowers each candidate's score depending on whether its first two ID codes appear in the user's listening profile.
6. **Explanations**: the LLM writes one sentence per result, using only the track's own fields.
7. **Feedback**: play, save and skip are stored in Postgres. Played and saved tracks join the profile, so the familiar side of the slider shifts over time.

## Stack

- Backend: FastAPI (`server.py`)
- Database: Postgres via `psycopg` (`db.py`): users, listens, searches, feedback
- Frontend: one static page (`static/index.html`), no build step
- LLM: OpenRouter when `OPENROUTER_API_KEY` is set, otherwise local Ollama (`qwen2.5:3b`) (`llm.py`)

## Run locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# download the dataset to data/dataset.csv first (see Dockerfile for the URL)
.venv/bin/python build_catalog.py   # catalog
.venv/bin/python embed.py           # embeddings
.venv/bin/python semantic_ids.py    # semantic IDs

# .env must define DATABASE_URL for a Postgres database
ollama pull qwen2.5:3b              # optional: local LLM
.venv/bin/uvicorn server:app --port 8000
```

Open http://localhost:8000. Without an LLM, search uses your words directly and explanations fall back to a template.

## Files

| File | Role |
| --- | --- |
| `build_catalog.py` | clean the CSV, sample 30k tracks, build item text |
| `embed.py` | embed item text |
| `semantic_ids.py` | residual k-means IDs and trie |
| `recommend.py` | trie-constrained beam search, profile, familiar/non-habitual switch |
| `llm.py` | query rewrite and explanations |
| `db.py` | Postgres storage |
| `server.py` | API and static hosting |
| `Dockerfile` | rebuilds the data at image build time |

## Limits

- Semantic IDs come from a pretrained embedder plus k-means. There is no model trained to generate them end to end, and no preference tuning (RLHF/DPO). Feedback is logged and updates the profile, but does not train anything.
- The user profile is one demo user seeded with 120 tracks. There are no accounts.
- Relevance leans on title and genre words more than on audio features, and many tracks fall into "neutral mood".
- Moving the slider to the familiar end costs some relevance.
- No automated evaluation yet (recall, novelty, explanation groundedness are planned in `MVP.md`).
- The 3B local model follows the output format only loosely; the hosted model is untested.
