import pandas as pd
import numpy as np

def data_load():
    dataset=pd.read_csv('news_dataset.csv')
    dataset= dataset.to_dict("records")
    return dataset
    