"""
本文件存放基础函数的实现
"""
import sys
sys.path.append("..")

from project1_quick_search.easykmap_tree import *
import pickle
import pandas as pd
from elasticsearch import Elasticsearch
import warnings
warnings.filterwarnings("ignore")
raw_data=pd.read_csv("../data/河南省证券交易所数据集.csv", encoding='gbk')
print("raw_data:",raw_data)
input_data_list=[(k,v) for k,v in zip(raw_data["证券交易所"], raw_data["历史检索热度"])]
print("input_data_list:",input_data_list)

#创建树
def build_tree(data_list):
    tree = build(data_list)
    f = open('./tree_file.npy','wb') #覆盖更新
    pickle.dump(tree, f) #持久化tree
    f.close()
    return tree
#实例化创建一个tree

tree=build_tree(data_list=input_data_list)

#通过tree词缀树查找相关匹配文档
def get_tree_search_result(tree,query,k):
    tree_search_result=[]
    for key, node in search(tree, u'{}'.format(query), limit=k):  # 按照node.weight进行倒排
        tree_search_result.append(key)
    return tree_search_result
# tree_search_result=get_tree_search_result(tree,query="平顶山",k=8)
# print("tree_search_result:",tree_search_result)

#基于ES进行倒排索引检索
def get_es_search_result(es, raw_finance_name):
    # 自定义内容,查询倒排索引匹配结果
    query = {
        "query": {
            "match": {
                "finance_name": {"query":raw_finance_name,"minimum_should_match": "10%"} #至少10%匹配才会被检索到
  # 全文搜索，支持分词
            }
        }
    }
    response = es.search(index="finance_search_index", body=query)
    if response:
        result_list=[k['_source']['finance_name'] for k in response['hits']['hits']] #取出对应检索到的value
    else:
        result_list=[]
    print("result_list:", result_list)  # 343990
    return result_list

from config.config import RagConfig
#连接ES,进行倒排索引检索
es_host=RagConfig().elastic_host
es_port = 9200
es = Elasticsearch(
    hosts=[{"host": es_host, "port": es_port, "scheme": "https"}],
    basic_auth=(RagConfig().elastic_username, RagConfig().elastic_password),
    verify_certs=False)
print("es:",es)
print(es.info())

es_search_result=get_es_search_result(es=es,raw_finance_name="平顶山")
print("es_search_result:",es_search_result)

#写一个检索优先级逻辑:输入一个用户输入的raw_finance_name片段,优先进行tree查找,然后如果不满8个进行es查找直到推荐数量达到8个
def get_final_recommend_result(es,tree,raw_finance_name,K=8):
    final_recommendation_list = get_tree_search_result(tree=tree,query=raw_finance_name,k=K)
    # 如果tree search的结果到达k=8个,那么直接返回推荐结果
    if len(final_recommendation_list) >= K:
        print(f"通过tree search方法获得推荐证券机构{len(final_recommendation_list)}个,推荐结果如下:{final_recommendation_list}")
        return final_recommendation_list[:K]
    #如果不足8个走es检索
    else:
        print(f"通过tree search方法获得推荐证券机构{len(final_recommendation_list)}个,推荐结果如下:{final_recommendation_list}")
        es_recommendation_list = get_es_search_result(es, raw_finance_name)
        remainder_slot = K - len(final_recommendation_list) #剩余槽位数量
        print(f"剩余{remainder_slot}个推荐位需要通过倒排索引填充")
        #将非重复结果填充进缺失的推荐位置中
        for elem in es_recommendation_list:
           if len(final_recommendation_list)>=remainder_slot:
               break
           if elem not in final_recommendation_list:
                final_recommendation_list.append(elem)
        return final_recommendation_list

# final_result=get_final_recommend_result(es=es,tree_search_result=tree_search_result,es_search_result=es_search_result,K=8)
# print("final_result:",final_result)