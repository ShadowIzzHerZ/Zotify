FROM python:3.12-slim
WORKDIR /app
# CPU-only torch keeps the image small
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# bake the embedding model into the image so startup does not download it
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
# rebuild catalog, embeddings and semantic IDs at build time (seeded, so reproducible);
# the deploy upload is limited to 28 MB, too small to ship data/
COPY build_catalog.py embed.py semantic_ids.py ./
RUN mkdir -p data && python -c "import urllib.request as u; u.urlretrieve('https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv', 'data/dataset.csv')" \
 && python build_catalog.py && python embed.py && python semantic_ids.py && rm data/dataset.csv
COPY . .
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}"]
