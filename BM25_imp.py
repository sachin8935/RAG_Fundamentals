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
    data_guid=[]
    for i in range(0,5):
        index= result[0][i]
        data_guid.append(dataset[index]['guid'])
    return data_guid
