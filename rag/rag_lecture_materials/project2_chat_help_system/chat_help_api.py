"""
证券公司名称匹配API接口实现
pip install fastapi
"""

from fastapi import FastAPI, Request,HTTPException, Depends,Query
import uvicorn
import json
from fastapi.middleware.cors import CORSMiddleware
from chat_help_main import *
import sys

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/rag_chat")
async def rag_chat_api(request: Request):
    try:
        raw_body = await request.body()
        print("raw_body:",raw_body)
        ipt = json.loads(raw_body.decode("utf-8"))
        print("ipt:",ipt)
        query=ipt.get("query","") #本轮用户会话
        history=ipt.get("history",[]) #[{"role":"user","content":"你好"},{"role":"assistant","content":"请问有什么能帮到您的呢？"}]
        print(query, history)
        response = answer_by_vectorstore(query=query, history=history)
        return {"response":response,"code":200,"msg":"success"}
    except Exception as e:
        print("出现错误")
        return {"response":"","code":500,"msg":f"service inner error:{e}"}

if __name__ == '__main__':
    http_address = "0.0.0.0"
    port=7069
    uvicorn.run(app="chat_help_api:app", host=http_address, port=port, workers=1)

# 服务后台启动命令
# nohup uvicorn chat_help_api:app --host 0.0.0.0 --port 7069 --workers 1  &
