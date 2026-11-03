"""Clean the raw tracks CSV and build one short text description per track."""
import pandas as pd

RAW = "data/dataset.csv"
OUT = "data/catalog.parquet"
N = 30_000


def band(x, words, cuts):
    return words[sum(x > c for c in cuts)]


def describe(r):
    parts = [
        band(r.energy, ["very low energy", "low energy", "medium energy", "high energy", "very high energy"], [.2, .4, .6, .8]),
        band(r.valence, ["dark and sad", "melancholic", "neutral mood", "upbeat", "very happy"], [.2, .4, .6, .8]),
        band(r.acousticness, ["electronic", "mostly produced", "mostly acoustic", "acoustic"], [.25, .5, .75]),
        band(r.tempo, ["slow", "mid tempo", "fast"], [90, 130]),
        "instrumental" if r.instrumentalness > 0.5 else "vocal",
        "danceable" if r.danceability > 0.7 else "",
    ]
    feats = ", ".join(p for p in parts if p)
    return f"{r.track_name} by {r.artists}. Genre: {r.track_genre}. {feats}."


def main():
    df = pd.read_csv(RAW)
    df = df.dropna(subset=["track_name", "artists"])
    df = df.sort_values("popularity", ascending=False)
    df = df.drop_duplicates("track_id").drop_duplicates(["track_name", "artists"])
    df = df.sample(n=min(N, len(df)), random_state=0).reset_index(drop=True)
    df["item_id"] = df.index
    df["text"] = df.apply(describe, axis=1)
    cols = ["item_id", "track_id", "track_name", "artists", "album_name", "track_genre",
            "popularity", "energy", "valence", "acousticness", "tempo", "danceability",
            "instrumentalness", "text"]
    df[cols].to_parquet(OUT, index=False)
    print(f"{len(df)} tracks -> {OUT}")
    print(df.text.head(3).to_string())


if __name__ == "__main__":
    main()
