"""
证券公司名称匹配API接口实现
"""

from logging import raiseExceptions
from fastapi import FastAPI, Request
import uvicorn
import json
from fastapi.middleware.cors import CORSMiddleware
from function_utils import *
import sys
sys.path.append("../")
from config.config import RagConfig

#创建全局树
raw_data=pd.read_csv("../data/河南省证券交易所数据集.csv",encoding='gbk')
print("raw_data:",raw_data)
input_data_list=[(k,v) for k,v in zip(raw_data["证券交易所"], raw_data["历史检索热度"])]
print("input_data_list:",input_data_list)
tree=build_tree(data_list=input_data_list)

#全局性操作
#初始状态先连接es(全局连接)
from elasticsearch import Elasticsearch
import warnings
warnings.filterwarnings("ignore")
es_host = RagConfig().elastic_host
es_port = 9200
es = Elasticsearch(
    hosts=[{"host": es_host, "port": es_port, "scheme": "https"}],
    basic_auth=(RagConfig().elastic_username, RagConfig().elastic_password),
    verify_certs=False)
print("es连接成功:", es)

K=8
#判读是否需要重新连接es的方法
def judge_to_reconnect(es=es):
    if es.ping() == False:
        es_host = RagConfig().elastic_host
        es_port = 9200
        es = Elasticsearch(
            hosts=[{"host": es_host, "port": es_port, "scheme": "https"}],
            basic_auth=(RagConfig().elastic_username, RagConfig().elastic_password),
            verify_certs=False)
        print("es连接成功:", es)
    return es

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/finance_recommendation")
async def bank_recommend_function(request: Request):
    try:
        es = judge_to_reconnect() #检查es连接是否失效,失效了重新连接
        raw_body = await request.body()
        print("raw_body:",raw_body)
        ipt = json.loads(raw_body.decode("utf-8"))
        print("ipt:",ipt)
        raw_finance_name=ipt.get('raw_finance_name','no_name')
        if raw_finance_name.strip()=="":
            return {"result":"请输入证券公司名称","code":200,"msg":"success"}
        print("输入的证券公司名称片段:",raw_finance_name)
        #存放最终的推荐证券公司名称的列表
        final_recommend_slots_list = get_final_recommend_result(es,tree,raw_finance_name,K=K)
        return {"result": final_recommend_slots_list,"code":200,"msg":"success"}
    except Exception as e:
        print("出现错误")
        return {"result":[],"code":500,"msg":f"service inner error:{e}"}

if __name__ == '__main__':
    http_address = "0.0.0.0"
    port=7067
    uvicorn.run(app="quick_search_api:app", host=http_address, port=port, workers=1)

# 服务后台启动命令
# nohup uvicorn quick_search_api:app --host 0.0.0.0 --port 7067 --workers 1  &
