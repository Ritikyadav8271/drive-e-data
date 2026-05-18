from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
import pandas as pd

resumes = [
    "Python SQL pandas data analysis machine learning statistics",
    "Java Spring Boot backend microservices",
    "Data analyst Python SQL Power BI data visualization",
    "Frontend developer HTML CSS JavaScript React"
]

job_description = "Looking for Data Analyst with Python SQL data analysis and visualization"

vectorizer = TfidfVectorizer(stop_words='english')

all_docs = resumes + [job_description]
tfidf_matrix = vectorizer.fit_transform(all_docs)

tfidf_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()



model = SentenceTransformer('all-MiniLM-L6-v2')

resume_embeddings = model.encode(resumes)
jd_embedding = model.encode([job_description])

embedding_scores = cosine_similarity(jd_embedding, resume_embeddings).flatten()


df = pd.DataFrame({
    "Resume": resumes,
    "TF-IDF Score": tfidf_scores,
    "Embedding Score": embedding_scores
})

df = df.sort_values(by="Embedding Score", ascending=False)

print(df)