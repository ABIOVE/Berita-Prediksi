import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("berita_HOAX_indonesia.csv", encoding="latin-1", sep=";")
df.columns = [c.replace("ï»¿", "") for c in df.columns]
df["berita"] = df["berita"].astype(str).str.strip('"')

# Label: valid = 1, hoax = 0
df["label"] = df["kategori"].map({"valid": 1, "hoax": 0})

# Pipeline TANPA stop_words
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2
    )),
    ("model", MultinomialNB())
])

# Train model
pipeline.fit(df["berita"], df["label"])

# Save model
joblib.dump(pipeline, "model_pipeline.pkl")

print("Model pipeline berhasil disimpan")
