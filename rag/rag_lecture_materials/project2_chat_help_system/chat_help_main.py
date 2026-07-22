"""
主体程序实现：
1. 对金融QA数据集1.csv以及对金融doc数据集1.txt 进行知识库融合构建
2.按照需求完成多层级的RAG整个过程
优先在metadata中type=query的子集合中按照阈值进行高相似度检索,看能否直接命中
"""
import sys
from build_knowledge_vectorstore import init_vectordb,get_bm25_retriever,generation_replay_by_LLM
#初始化向量库
vectorstore=init_vectordb()
high_resolution_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.82},
)

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
#BM25检索器:用于在中精度和自由生成时提供一定的其它召回机制的支持
bm25_retriever=get_bm25_retriever(K=5)

#定义主函数,用于使用三个level的检索器解答用户提出的问题query
def answer_by_vectorstore(query,
                          history,
                          query_retriever=high_resolution_retriever,
                          qa_retriever=middle_resolution_retriever,
                          doc_retriever=doc_retriever,
                          bm25_retriever=bm25_retriever):
    """
    :param query:用户本轮问题
    :param history: 历史记录
    :param query_retriever:基于query的检索器
    :param qa_retriever: 基于qa的检索器
    :param doc_retriever: 文档检索器
    :param bm25_retriever: 稀疏bm25检索器
    :return: answer
    """
    high_resolution_docs_list = query_retriever.invoke(query, filter={"type": "query"}) #通过添加filer来过滤查找范围为query标签的内容
    print("high_resolution_docs_list:",high_resolution_docs_list)
    #如果获取到了刚精度匹配的结果,说明直接可以answer回复用户,无需经过LLM费时间的生成方式
    if high_resolution_docs_list:
        # 取首个(最相关)的文档作为答案直接回复用户问题
        return [k.metadata["answer"] for k in high_resolution_docs_list][0]
    #如果高精度query没有匹配上,尝试用qa整体去和用户的query做匹配(这样能最大限度的利用query+answer)的知识
    middle_resolution_docs_list=qa_retriever.invoke(query, filter={"type": "qa"})
    print("middle_resolution_docs_list:",middle_resolution_docs_list)
    bm25_retriever_result=bm25_retriever.invoke(query) #单独建立的稀疏向量结构,直接检索即可
    print("bm25_retriever_result:",bm25_retriever_result)
    #如果有命中中精度的匹配结果，追加通过bm25进行获取5个文档一起交给模型进行生成
    combine_knowledge_str=""
    #如果有qa检索结果直接和bm25检索结果合并作为backgroup_info
    if middle_resolution_docs_list:
        for k in middle_resolution_docs_list:
            #将知识按照 问题：xxx 答案：xxx
            #         问题:xxx  答案:xxx 的格式进行组织
            print("k:",k)
            combine_knowledge_str+="问题:"+k.page_content.split("答案：")[0]+"答案："+k.page_content.split("答案：")[1]+"\n"
        print("qa_knowledge_combine_str:",combine_knowledge_str)
        for k in bm25_retriever_result:
            combine_knowledge_str+="文档查阅结果:"+k.page_content+"\n"
        print("total_combine_str:",combine_knowledge_str)
    else: #如果没有qa检索结果,再进行doc_retriever检索,合并bm25检索结果作为backgroup_info
        doc_retriever_result = doc_retriever.invoke(query)
        print("doc_retriever_result:",doc_retriever_result)
        if doc_retriever_result:
            for doc in doc_retriever_result:
                combine_knowledge_str="文档查阅结果:"+doc.page_content+"\n"

        for k in bm25_retriever_result:
            combine_knowledge_str += "文档查阅结果:" + k.page_content + "\n"
        print("total_combine_str:", combine_knowledge_str)

    #对历史记录的处理：拼接为统一的字符串
    chat_history_combine_str=""
    for his in history:
        if his["role"]=="user":
            chat_history_combine_str+="用户:"+his["content"]+"\n"
        else:
            chat_history_combine_str+="AI:"+his["content"]+"\n"
    print("chat_history_combine_str:", chat_history_combine_str)
    #调用LLM基于知识检索结果、历史记录进行QA问答
    response = generation_replay_by_LLM(background_knowledge=combine_knowledge_str,
                                        history_message=chat_history_combine_str,
                                        current_query=query)
    return response

# query="第2季度GDP增速2.4%但贫富分化加剧基尼系数是多少"
# query="你好"
# query="什么是股票期权？" #股票期权是一种赋予持有者在未来某一日期以特定价格买卖股票的权利的金融衍生品。
# history=[{"role":"user","content":"你好"},{"role":"assistant","content":"请问有什么能帮到您的呢？"}]
# response=answer_by_vectorstore(query=query,history=history)
# print("response:", response)