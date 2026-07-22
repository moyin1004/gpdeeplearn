"""
本文件用于测试向量数据库接口:
vectorstore_api(homework_answer).py中的接口是否能正常工作，模拟请求
"""
#%%
import requests
# #type:insert/update/search/delete  #dtype:qa/query/docs
#插入一条q - a 型数据
json_data={"type":"insert","dtype":"qa","query":"我今天心情很好","answer":"太棒了，我们一起出去玩吧"}
response=requests.post(url="http://xxx.xxx.xxx.xxx:6666/vectorstore",json=json_data)
print("response:",response.json())
#%%
#查询刚插入的数据
json_data={"type":"search","dtype":"qa","query":"我今天心情很好，太棒了，我们一起出去玩吧"}
response=requests.post(url="http://xxx.xxx.xxx.xxx:6666/vectorstore",json=json_data)
print("response:",response.json())
#%%
#删除数据
json_data={"type":"delete","dtype":"qa","query":"我今天心情很好答案：太棒了，我们一起出去玩吧"}
response=requests.post(url="http://xxx.xxx.xxx.xxx:6666/vectorstore",json=json_data)
print("response:",response.json())
#%%
#再次查询这条数据,发现找不到这条数据啦
json_data={"type":"search","dtype":"qa","query":"我今天心情很好，太棒了，我们一起出去玩吧"}
response=requests.post(url="http://xxx.xxx.xxx.xxx:6666/vectorstore",json=json_data)
print("response:",response.json())

#剩余场景自己测试效果,修改代码吧