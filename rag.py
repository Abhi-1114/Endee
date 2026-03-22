from embeddings import get_embedding
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def retrieve(query, docs, embeddings):
    query_vec = get_embedding(query)
    scores = cosine_similarity([query_vec], embeddings)[0]
    best_idx = np.argmax(scores)
    return docs[best_idx]

def answer(query, docs, embeddings):
    context = retrieve(query, docs, embeddings)
    return f"Answer based on context: {context}"
