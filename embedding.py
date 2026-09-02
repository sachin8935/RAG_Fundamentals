from openai import OpenAI
import data_filteration
import numpy as np
client= OpenAI()
def semantic_embed_dataset():
    dataset_guid_embed=[]
    dataset_embedding=[]
    dataset= data_filteration.data_load()
    filtered_data_title=[]
    for word in dataset:
        filtered_data_title.append(word['title'])
    response = client.embeddings.create(
    input= filtered_data_title,
    model="text-embedding-3-small"
    )
    for i in range(0,len(filtered_data_title)):
        dataset_guid_embed.append(dataset[i]['guid'])
        dataset_embedding.append(response.data[i].embedding)
    np.savez_compressed(
        "dataset_embedding.npz",
        guids=np.array(dataset_guid_embed),
        embeddings=np.array(dataset_embedding)
        )