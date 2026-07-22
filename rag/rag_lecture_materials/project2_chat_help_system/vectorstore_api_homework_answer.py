#%%

# 写一个向量接口服务，功能：
#接收用户请求根据操作类型type决定干什么
# type in search/add/delete/update
from fastapi import FastAPI, Request,HTTPException, Depends,Query
import uvicorn
import json
from fastapi.middleware.cors import CORSMiddleware
import sys
sys.path.append("./project2_chat_help_system")

from chat_help_main import *

from config import rag_test_Config
import pandas as pd

#%%

#%%



import sys
# sys.path.append("./project2_chat_help_system")

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
    embeddings = DashScopeEmbeddings(model="text-embedding-v2",dashscope_api_key=rag_test_Config().api_key)
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
    manipulate_type=ipt.get("type","")  #操作类型:insert/update/search/delete
    data_type=ipt.get("dtype","") #知识类型data_type: qa/query/docs
    query=ipt.get("query","") #本轮用户会话
    answer=ipt.get("answer","")
    # 如果操作类型是插入向量数据库数据
    if manipulate_type=="insert":
        if data_type=="query": #根据操作类型决定插入哪一类知识
            vectorstore.add_documents([Document(metadata={'type': 'query', 'ids':str(uuid.uuid4()),"answer":answer},
                                                page_content=query)])
        elif data_type=="qa":
            vectorstore.add_documents([Document(metadata={'type': 'qa', 'ids':str(uuid.uuid4())},
                                                page_content=query+"答案："+answer)])
        elif data_type=="docs":
            vectorstore.add_documents([Document(metadata={'type': 'docs', 'ids':str(uuid.uuid4())},
                                                page_content=query)])
        else:
            return {"docs": [], "message": "无效的操作类型,向量插入失败"}
        return {"docs": [], "message": "向量插入操作成功"}
    #如果操作类型是：查询向量,根据不同操作类型进行向量检索
    elif manipulate_type=="search":
        if data_type=="query":
            docs = retriever.get_relevant_documents(query,filter={"type": "query"})  # 通过添加filer来过滤查找范围为query标签的内容
        elif data_type=="qa":
            docs = qa_retriever.get_relevant_documents(query,filter={"type": "qa"})  # 通过添加filer来过滤查找范围为query标签的内容
        elif data_type=="docs":
            docs=doc_retriever.get_relevant_documents(query,filter={"type": "docs"})
        else:
            return {"docs": [], "message": "无效的检索模式,向量检索失败"}
        if docs:
            return {"docs":[k.page_content for k in docs],"message":"向量检索操作成功"}
        else:
            return {"docs":[],"message":"向量检索操作成功"}
    #如果操作类型是更新向量知识库:那么先根据检索模式删除知识，再插入新的知识
    elif manipulate_type=="update":
        #先删除知识
        if data_type=="query": #如果要删除的是query类型的知识
            #获取query类型
            result=retriever.get_relevant_documents(query)
        elif data_type=="qa": #如果要删除的是qa类型的某条知识
            result=qa_retriever.get_relevant_documents(query)
        elif data_type=="docs": #如果要删除的是docs类型的某条知识
            result=doc_retriever.get_relevant_documents(query)
        else:
            return {"doc": [], "message": "无效的检索模式,向量删除失败"}
        #找到我们指定的metadata中的ids,取最相近的一个
        temp_list=[k.metadata["ids"] for k in result]
        if temp_list:
            object_id=temp_list[0]
            #通过metadata中的ids找到对应的系统ids
            delete_query = vectorstore.get(where={"ids": object_id})
            print("需要删除的系统ids:",delete_query)
            #将系统的ids对应的向量记录删除
            if delete_query["ids"]:
                vectorstore.delete(ids=delete_query["ids"]) #删除系统ids
        #再重新插入知识
        if data_type=="query":
            vectorstore.add_documents([Document(metadata={'type': 'query', 'ids':str(uuid.uuid4()),"answer":answer},
                                                page_content=query)])
        elif data_type=="qa":
            vectorstore.add_documents([Document(metadata={'type': 'qa', 'ids':str(uuid.uuid4())},
                                                page_content=query+"答案："+answer)])
        elif data_type=="docs":
            vectorstore.add_documents([Document(metadata={'type': 'docs', 'ids':str(uuid.uuid4())},
                                                page_content=query)])
        else:
            return {"docs": [], "message": "无效的检索模式,向量更新失败"}

        return {"docs": [], "message": "向量删除操作成功"}


    #如果操作类型是删除向量知识库中的某条值为data_type类型的query
    elif manipulate_type=="delete":
        if data_type=="query": #如果要删除的是query类型的知识
            #获取query类型
            result=retriever.get_relevant_documents(query)
        elif data_type=="qa": #如果要删除的是qa类型的某条知识
            result=qa_retriever.get_relevant_documents(query)
        elif data_type=="docs": #如果要删除的是docs类型的某条知识
            result=doc_retriever.get_relevant_documents(query)

            #获取对应的metadata中的id
        else:
            return {"doc": [], "message": "无效的检索模式,向量删除失败"}
        # 找到对应的ids,取最相近的一个
        object_id = ""
        temp_list = [k.metadata["ids"] for k in result]
        if temp_list:
            object_id=temp_list[0]
            #通过metadata中的ids找到对应的系统ids
            delete_query = vectorstore.get(where={"ids": object_id})
            print("需要删除的系统ids:",delete_query)
            #将系统的ids对应的向量记录删除
            if delete_query["ids"]:
                vectorstore.delete(ids=delete_query["ids"])
                return {"docs": [], "message": "向量删除操作成功"}
            else:
                return {"docs": [], "message": "找不到对应的系统ids"}
        else:
            return {"docs": [], "message": "无匹配记录，删除失败"}
    else: #无效操作类型
        return {"doc":[],"message":"无效操作类型，操作不成功"}


if __name__ == '__main__':
    http_address = "0.0.0.0"
    port=6666
    uvicorn.run(app="vectorstore_api(homework_answer):app", host=http_address, port=port, workers=1)

    # 服务后台启动命令
    # nohup uvicorn vectorstore_api_homework_answer:app --host 0.0.0.0 --port 6666 --workers 1  &
