import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

METADATA = "metadata/X-ray_Metadata - Sheet1.csv"

class TextSearch:
    def __init__(self):
        self.df = pd.read_csv(METADATA)
        self.df["text"] = self.df["category"].str.lower()
        self.vectorizer = TfidfVectorizer()
        self.tfidf = self.vectorizer.fit_transform(self.df["text"])

    def search(self, query, top_k=5):
        q_vec = self.vectorizer.transform([query.lower()])
        sims = cosine_similarity(q_vec, self.tfidf)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        return self.df.iloc[top_idx][["image_name", "category"]]
