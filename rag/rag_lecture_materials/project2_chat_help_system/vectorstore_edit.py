"""
向量数据库的CRUD操作
"""
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

#%%
"""
创建统一知识库,会有三类知识：
文档型知识->文档分块知识
QA型知识:->(1)基于query高精度匹配的知识 (2)基于qa中精度匹配的知识
需要以统一的格式写入同一个知识库,然后通过meta_datas中指定不同类型的type进行区分
"""
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
    embeddings = DashScopeEmbeddings(model="text-embedding-v2",dashscope_api_key=rag_test_Config.api_key) #
    print(embeddings)
    vectorstore = Chroma.from_documents(combine_document_list, embeddings,collection_name="novel"  # 指定持久化目录,persist_directory="./db1"
) #, persist_directory="db
    # 然后你就可以像平常一样使用vectorstore了
    return vectorstore
vectorstore=init_vectordb(qa_raw_data)

#%%
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.1},
)
query="z"
docs = retriever.get_relevant_documents(query, filter={"type": "query"}) #通过添加filer来过滤查找范围为query标签的内容
print("docs:",[k.metadata["ids"] for k in docs])
#%%
#向量数据库的删除操作,删除id为1的向量数据 [ids]属性:它是系统创建向量时默认创建的属性,不能人工赋值
# existing = vectorstore.get(where={"ids":"5fd9f02f-acc6-49c2-9200-7b1c94654cca"})
existing_id=vectorstore.get(where={"answer":"股票的内在价值是指根据未来现金流折现计算的股票理论价值，通常用于评估股票的合理价格。"})
print(existing_id)
# ['1aab34d3-27ab-4ae1-8eac-f7b5a3cb95f8', 'f83770d8-eb76-4a36-ba08-4fce49bb791f'] #系统创建的ids
vectorstore.delete(ids=['1aab34d3-27ab-4ae1-8eac-f7b5a3cb95f8','f83770d8-eb76-4a36-ba08-4fce49bb791f'])
#%%

existing = vectorstore.get(where={"ids":"5fd9f02f-acc6-49c2-9200-7b1c94654cca"})
print("existing:",existing)

retriever1 = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.3},
)
docs = retriever1.get_relevant_documents(query) #通过添加filer来过滤查找范围为query标签的内容
print("docs:",docs)
#%%
existing = vectorstore.get(where={"type":"docs"})
print("Existing before delete:", existing)
#%%
#向量数据库的新增操作
vectorstore.add_documents([Document(metadata={'type': 'query', 'ids': '234',"answer":"2222"}, page_content='11111')])
existing = vectorstore.get(where={"ids":"234"})
print(existing)
#%%
#基于元数据metadata查找到某类输出进行删除
result=vectorstore.get(where={"type":"docs"})
#如果result中记录则进行删除
if result['ids']:
    vectorstore.delete(result["ids"])












