"""知识库构建代码"""

#导入QA型知识库
import pandas as pd
import sys
sys.path.append("../")
from config.config import RagConfig

qa_raw_data=pd.read_csv("../data/金融QA数据集.csv", encoding="GB2312")
print(qa_raw_data)
print(qa_raw_data["Question"])

"""
创建统一知识库,会有三类知识：
文档型知识->文档分块知识
QA型知识:->(1)基于query高精度匹配的知识 (2)基于qa中精度匹配的知识
需要以统一的格式写入同一个知识库,然后通过meta_datas中指定不同类型的type进行区分
"""
from typing import List, Optional, Dict, Any
import uuid
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1.创建query型知识在向量库中的数据结构,需要将answer放入metadatas中
def init_vectordb(qa_raw_data=qa_raw_data,api_key=RagConfig().api_key):
    query_list_document_list=[]
    raw_query_list_document_list = [Document(page_content=q) for q in qa_raw_data["Question"]]
    for i,doc in enumerate(raw_query_list_document_list):
        doc.metadata["type"]="query"
        doc.metadata["answer"]=qa_raw_data["Answer"][i]
        doc.metadata["id"]=str(uuid.uuid4()) #增加上唯一id，这个id有啥用后续会讲
        query_list_document_list.append(doc)
    print("query_list_document_list:",query_list_document_list)

    #2.创建qa型知识在向量库中的数据结构，需要将问题和答案拼接,通过答案：进行标识
    qa_list_document_list=[]
    raw_qa_list_document_list = [Document(page_content=q+"答案："+a) for q,a in zip(qa_raw_data["Question"], str(qa_raw_data["Answer"]))]
    for i, doc in enumerate(raw_qa_list_document_list):
        doc.metadata["type"]="qa"
        doc.metadata["id"]=str(uuid.uuid4()) #增加上唯一id，这个id有啥用后续会讲
        qa_list_document_list.append(doc)
    print("qa_list_document_list:",qa_list_document_list)

    #3.长文档建立向量库的元数据
    loader = TextLoader("../data/文档型知识.txt")
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
        doc.metadata["id"]=str(uuid.uuid4()) #增加上唯一id，这个id有啥用后续会讲
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
    # embeddings = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=api_key)
    embeddings = OpenAIEmbeddings(model="Pro/BAAI/bge-m3", base_url="https://api.siliconflow.cn/v1",
                                  api_key=RagConfig.api_key)
    embeddings_success = False  # sym:embeddings 是否成功的判断
    try:
        test_embedding = embeddings.embed_documents(["测试连接是否成功"])
        embeddings_success = bool(test_embedding and len(test_embedding) > 0)
    except Exception as exc:
        print("embeddings connection or generation failed:", repr(exc))
    print("embedding model is ", embeddings)
    print("embeddings success:", embeddings_success)
    vectorstore = Chroma.from_documents(combine_document_list, embeddings, collection_name="novel") #, persist_directory="db
    # 然后你就可以像平常一样使用vectorstore了
    return vectorstore

vectorstore=init_vectordb(qa_raw_data)
"""
构建三个不同score_threshold(检索精度)的检索器:
high_resolution_retriever(高精度检索器):score_threshold=0.82 用于匹配精确命中type=query 类的知识
middle_resolution_retriever(中等精度检索器):score_threshold=0.68 用于匹配与type=qa相关的所有内容
doc_retriver(文档型检索器)：用于匹配type=docs相关的内容，score_threshold=0.65作为前两类检索器都没有检索到结果时候的补充检索
"""
high_resolution_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.82},
)
# query="解释Black-Scholes期权定价模型的核心假设"
# docs = high_resolution_retriever.get_relevant_documents(query, filter={"type": "query"}) #通过添加filer来过滤查找范围为query标签的内容
# print("docs:",docs)

middle_resolution_retriever=vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.68},
)
# query="解释Black-Scholes期权定价模型的核心假设"
# docs = middle_resolution_retriever.get_relevant_documents(query, filter={"type": "qa"}) #通过添加filer来过滤查找范围为qa标签的内容
# print("docs:",docs)

doc_retriever=vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.3},
)
# query="n数据悖论：Q2 GDP增速2.4%但贫富分化加剧基尼系数0.49是什么原因"
# docs = doc_retriever.get_relevant_documents(query, filter={"type": "docs"}) #通过添加filer来过滤查找范围为qa标签的内容
# print("docs:",docs)

# query="n数据悖论：Q2 GDP增速2.4%但贫富分化加剧（基尼系数0.49）"
# docs = doc_retriever.get_relevant_documents(query, filter={"type": "docs"}) #通过添加filer来过滤查找范围为qa标签的内容
# print("docs:",docs)
# print([k.page_content for k in docs])

#使用bm25retriever进行检索
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import jieba
def chinese_tokenizer(text: str):
    tokens = jieba.lcut(text)
    return [token for token in tokens] # if token not in stopwords.words('chinese')
print(chinese_tokenizer("我今天去吃火锅了，火锅非常好吃"))

#---------- (5) 构建bm25 retriever器 -------------
#对于bm25类的retriever我们将qa类型的数据和文档类型的数据进行拼接同步进行检索
def get_bm25_retriever(K): #默认分词器基于英文空格作为拆分符号进行分词的
    bm25_doc_list = []
    # 追加上QA类型知识
    bm25_doc_list.extend([q + "答案：" + a for q, a in zip(qa_raw_data["Question"], str(qa_raw_data["Answer"]))])
    # 追加上文档类型知识
    loader = TextLoader("../data/文档型知识.txt")
    doc_documents = loader.load_and_split()
    print("documents:", doc_documents)
    # 2. 文本分割（按段落/句子切分）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=30,  # 每个分块的最大字符数，设置的小一些这里
        chunk_overlap=5,  # 分块之间的重叠字符数
        separators=["\n\n", "\n", "。", "？", "！", "；"]  # 中文分割符
    )
    # 进行文档分割
    splits_documents = text_splitter.split_documents(doc_documents)
    doc_bm25_list = [k.page_content for k in splits_documents]
    bm25_doc_list.extend(doc_bm25_list)
    print("bm25_doc_list:", bm25_doc_list)
    # 使用示例
    bm25_retriever = BM25Retriever.from_texts( #Create a BM25Retriever from a list of texts.
        bm25_doc_list, metadatas=[{"type": "bm25"}] * len(bm25_doc_list),
        preprocess_func=chinese_tokenizer, #自定义分词器
    )
    bm25_retriever.k = K #设置检索条目数=2
    #使用Bm25检索器对文本进行检索
    return bm25_retriever

#根据提供的背景检索材料以及LLM自由生成答案
from langchain_community.llms import openai  # 通义千问的LangChain集成
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain

# 初始化模型
llm = openai.OpenAI (
    model = "deepseek-ai/DeepSeek-V3.2",  # 使用qwen-max模型
    temperature=0.7,        # 控制随机性 (0~1)
    top_p=0.9,              # 控制多样性 (0~1)
    api_key=RagConfig().api_key,
    base_url="https://api.siliconflow.cn/v1"
)
print("llm:",llm)

#调用LLMChain获取对应的结果
def generation_replay_by_LLM(background_knowledge,history_message,current_query,llm=llm):
    """
    :param background_knowledge: 通过检索得到的背景信息字符串
    :param history_message: 上下文历史记录，这里会组合成一个字符串
    :param current_query: 当前用户问题
    :return:
    """
    prompt_template = """你是一个人工客服,名叫小丽,你能用礼貌的态度，基于背景信息回答用户的问题。如果背景信息与用户的问题相关则需要结合背景信息进行回答(不能乱编)，
        如果背景信息与用户的问题无关则忽略背景信息，直接回答。
        背景信息:{background_knowledge}
        历史记录:{history_message}
        用户问题:{current_query}
        回答："""
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["background_knowledge", "history_message", "current_query"])
    # print(prompt.format())
    # 创建LLM Chain
    chain = LLMChain(llm=llm, prompt=prompt)
    # print(chain)
    # 调用模型并打印结果
    response = chain.invoke({"current_query": current_query, "history_message": history_message,
                          "background_knowledge": background_knowledge})
    return response