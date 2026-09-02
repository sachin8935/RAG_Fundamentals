import bm25s
import data_filteration

def data_retrieval_BM(user_query,top_k):
    dataset= data_filteration.data_load()

    filtered_data=[]
    for data in dataset:
        filtered_data.append(data['title'])

    tokenized_data = bm25s.tokenize(filtered_data)
    retrieval = bm25s.BM25()
    retrieval.index(tokenized_data)

    query=user_query
    tokenized_query = bm25s.tokenize(query)

    result,scores = retrieval.retrieve(
    tokenized_query,
    k=top_k
    )

    return result,scores
