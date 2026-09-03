import numpy
import pandas as pd
import BM25_imp
import semantic
from dotenv import load_dotenv
load_dotenv()
def give_count(data_guid,result):
    cnt=0
    for i in range(0,len(data_guid)):
        for j in range(0,len(result)):
            if data_guid[i] == result[j]:
                cnt+=1
    return cnt

def calculate_precision(matching, total_retrieved):
    return (matching/total_retrieved)*100

def calculate_recall(matching,total_retrieved):
    return (matching/total_retrieved)*100

def print_recall(data):
    total=0
    for i in range(len(data)):
        total+=data[i]
        print(f"\nThe recall value for {i+1}th query is {data[i]}")
    result = total/len(data)
    print(f"\nThe overall recall value for data is {result}")


def print_precision(data):
    total=0
    for i in range(len(data)):
        total+=data[i]
        print(f"\nThe precision value for {i+1}th query is {data[i]}")
    result = total/len(data)
    print(f"\nThe overall precision value for data is {result}")


def get_precison_recall():
    data = pd.read_json('user_queries.json')
    keyword_accuracy_precision=[]
    semantic_accuaracy_precision=[]
    keyword_accuracy_recall=[]
    semantic_accuaracy_recall=[]
    for i in range(0,len(data)):
        query = data['query'][i]
        keyword_search = BM25_imp.data_retrieval_BM(query,5)
        semantic_search = semantic.semantic_results(query,5)
        relevent_guid = data['relevant_guids'][i]
        keyword_accuracy_cnt=give_count(keyword_search,relevent_guid)
        semantic_accuaracy_cnt=give_count(semantic_search,relevent_guid)
        keyword_accuracy_precision.append(calculate_precision(keyword_accuracy_cnt,len(keyword_search)))
        keyword_accuracy_recall.append(calculate_recall(keyword_accuracy_cnt,len(relevent_guid)))
        semantic_accuaracy_precision.append(calculate_precision(semantic_accuaracy_cnt,len(semantic_search)))
        semantic_accuaracy_recall.append(calculate_recall(semantic_accuaracy_cnt,len(relevent_guid)))

    print("\nThe data for keyword Search is as follows")
    print_precision(keyword_accuracy_precision)
    print_recall(keyword_accuracy_recall)
    print("\nThe data for Semantic Search is as follows")
    print_precision(semantic_accuaracy_precision)
    print_recall(semantic_accuaracy_recall)

get_precison_recall()