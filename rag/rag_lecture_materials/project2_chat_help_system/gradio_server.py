"""
本文件用于搭建一个简易的可以传递历史记录的前端页面
pip install gradio
"""
import gradio as gr
import asyncio
from typing import AsyncGenerator, List, Tuple
from chat_help_main import answer_by_vectorstore
#接受gradio前端传递的信息->进行消息转换,调用后端消息服务处理,完成消息处理后流式输出
async def gradio_wrapper(message: str, chat_history : List[List[str]]) -> AsyncGenerator[str, None]:
    """
    Gradio适配器，将Gradio格式的历史记录转换为chat_round需要的格式
    message: 当前用户消息
    history: Gradio格式的历史记录 [[用户消息1, 机器人回复1], [用户消息2, 机器人回复2], ...]
    """
    # 转换历史记录格式
    print("chat_history:",chat_history)
    converted_history1 = [(h[0], h[1]) for h in chat_history] if chat_history  else []
    converted_history=[]
    if converted_history1:
        for elem in converted_history1:
            converted_history.append({"role":"user","content":elem[0]})
            converted_history.append({"role":"assistant","content":elem[1]})
    # 调用你的对话函数
    response=answer_by_vectorstore(query=message, history=converted_history)
    return response

# 创建聊天界面
demo = gr.ChatInterface(
    fn=gradio_wrapper,
    title="金融智能咨询实战",
    description="欢迎咨询金融客服小丽",
    chatbot=gr.Chatbot(
        bubble_full_width=False,
        avatar_images=(
            "https://example.com/user.png",  # 用户头像URL
            "https://example.com/bot.png"  # 机器人头像URL
        ),
        height=500,
        render_markdown=True
    ),
    textbox=gr.Textbox(
        placeholder="请输入您的问题...",
        container=False,
        scale=7,
        autofocus=True
    ),
    # retry_btn="重试",
    # undo_btn="撤销",
    # clear_btn="清空"
)
# 启动应用
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", #任务其他服务可以访问该服务
        server_port=7860, #外网端口,端口
        share=True,#如果需要外网访问需要打开，网址:服务器公网ip地址(121.64.234.36..):端口号(7860)
    )