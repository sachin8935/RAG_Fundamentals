import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()
client= OpenAI()

def semantic_results(query,top_k):
    data = np.load("dataset_embedding.npz")
    response = client.embeddings.create(
    input=query,
    model="text-embedding-3-small"
    )
    data = np.load("dataset_embedding.npz")
    query_embedding=response.data[0].embedding
    similarity = cosine_similarity(
    [query_embedding],
    data['embeddings']
    )[0]
    similarity = enumerate(similarity)
    similarity= sorted(similarity,key=lambda x:x[1],reverse=True)
    result=[]
    for i in range(0,top_k):
        index,probability = similarity[i]
        result.append(data['guids'][index])
    return result
