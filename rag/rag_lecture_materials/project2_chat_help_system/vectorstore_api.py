#%%

# 写一个向量接口服务，功能：
#接收用户请求根据操作类型type决定干什么
# type in search/add/delete/update
from fastapi import FastAPI, Request,HTTPException, Depends,Query
import uvicorn
import json
from fastapi.middleware.cors import CORSMiddleware
from chat_help_main import *
import sys
sys.path.append("./project2_chat_help_system")

#%%

#%%
import uuid
import pandas as pd

import config

# from pygments.styles.dracula import background
import sys
sys.path.append("./project2_chat_help_system")
from config import rag_test_Config

qa_raw_data=pd.read_csv("/root/wangshihang/rag_lecture_materials/金融QA数据集1.csv")
print(qa_raw_data)
#%%
print(qa_raw_data["Question"])
#%%
#导入文档型知识库
doc_raw_data=pd.read_csv("/root/wangshihang/rag_lecture_materials/文档型知识.txt")
print(doc_raw_data)

from typing import List, Optional, Dict, Any
import uuid
import chromadb
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

    #1.创建query型知识在向量库中的数据结构,需要将answer放入metadatas中
def init_vectordb(qa_raw_data=qa_raw_data):
    query_list_document_list=[]
    raw_query_list_document_list = [Document(page_content=q) for q in qa_raw_data["Question"]]
    for i,doc in enumerate(raw_query_list_document_list):
        doc.metadata["type"]="query"
        doc.metadata["ids"]=str(uuid.uuid4()) #增加上唯一id，这个id有啥用后续会讲
        query_list_document_list.append(doc)
    print("query_list_document_list:",query_list_document_list)

    #2.创建qa型知识在向量库中的数据结构，需要将问题和答案拼接,通过答案：进行标识
    qa_list_document_list=[]
    raw_qa_list_document_list = [Document(page_content=q+"答案："+a) for q,a in zip(qa_raw_data["Question"], qa_raw_data["Answer"])]
    for i,doc in enumerate(raw_qa_list_document_list):
        doc.metadata["type"]="qa"
        doc.metadata["ids"]=str(uuid.uuid4()) #增加上唯一id，这个id有啥用后续会讲
        qa_list_document_list.append(doc)
    print("qa_list_document_list:",qa_list_document_list)
    #3.长文档建立向量库的元数据
    loader = TextLoader("/root/wangshihang/rag_lecture_materials/文档型知识.txt")
    doc_documents = loader.load_and_split()
    print("documents:",doc_documents)
    # 2. 文本分割（按段落/句子切分）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=30,      # 每个分块的最大字符数，设置的小一些这里
        chunk_overlap=5,   # 分块之间的重叠字符数
        separators=["\n\n", "\n", "。", "？", "！", "；"]  # 中文分割符
    )
    #进行文档分割
    splits_documents = text_splitter.split_documents(doc_documents)
    print(splits_documents)
    #对documents的元数据做一下修改增加type类型
    doc_list_document_list = []
    for i,doc in enumerate(splits_documents):
        doc.metadata["type"]="docs"
        doc.metadata["ids"]=str(uuid.uuid4()) #增加上唯一id，这个id有啥用后续会讲
        doc_list_document_list.append(doc)
    print("doc_list_document_list:",doc_list_document_list)
    #将三类知识合并成在一起,这个combine_document_list就是后续构建向量库用的统一数据结构了
    combine_document_list=query_list_document_list
    combine_document_list.extend(qa_list_document_list)
    combine_document_list.extend(doc_list_document_list)
    print("combine_document_list:",combine_document_list)
    print(len(combine_document_list)) #208=
    #构建向量库
    #载入向量模型
    embeddings = DashScopeEmbeddings(model="text-embedding-v2",dashscope_api_key=rag_test_Config.api_key)
    print(embeddings)
    vectorstore = Chroma.from_documents(combine_document_list, embeddings,collection_name="novel"  # 指定持久化目录,persist_directory="./db1"
) #, persist_directory="db
    # 然后你就可以像平常一样使用vectorstore了
    return vectorstore
vectorstore=init_vectordb(qa_raw_data)
retriever= vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.85},
)

qa_retriever= vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.6},
)
doc_retriever= vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.6},
)


#%%
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/vectorstore")
async def vector_api(request: Request):
    raw_body = await request.body()
    print("raw_body:",raw_body)
    ipt = json.loads(raw_body.decode("utf-8"))
    print("ipt:",ipt)
    manipulate_type=ipt.get("type","")  #操作类型
    # data_type=ipt.get("dtype","") #qa/query/doc
    query=ipt.get("query","") #本轮用户会话
    answer=ipt.get("answer","")
    if manipulate_type=="insert":
        if dtype=="query":
            vectorstore.add_documents([Document(metadata={'type': 'query', 'ids':str(uuid.uuid4()),"answer":answer},
                                                page_content=query)])
        elif dtype=="qa":
            vectorstore.add_documents([Document(metadata={'type': 'qa', 'ids':str(uuid.uuid4())},
                                                page_content=query+"答案："+answer)])
        else:
            ...
    elif manipulate_type=="search":
        docs = retriever.get_relevant_documents(query)  # 通过添加filer来过滤查找范围为query标签的内容
        # if data_type=="qa":
        #     docs = qa_retriever.get_relevant_documents(query)  # 通过添加filer来过滤查找范围为query标签的内容

        return {"docs":docs}
    elif manipulate_type=="update":
        pass #自己补充完整
    elif manipulate_type=="delete":
        result=vectorstore.get_relevant_documents(query)
        #先获取到对应的结果,如果有结果
        if result.get('ids',[]):
            vectorstore.delete(result["ids"][0])
        return {"docs":""}

if __name__ == '__main__':
    http_address = "0.0.0.0"
    port=7088
    uvicorn.run(app="vectorstore_api:app", host=http_address, port=port, workers=1)