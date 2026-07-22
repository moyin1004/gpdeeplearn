"""
测试接口返回结果
"""

import sys
sys.path.append("../")
import requests
from config.config import RagConfig
demo_url = f"http://{RagConfig().server_host}:7067/finance_recommendation" #112.124.57.93:6422 8081 http://192.168.7.140:7777/bank_recommendation
_data = {
    "raw_finance_name":"山", #平顶山 河南 你好
   }
test_response = requests.post(demo_url, json=_data,timeout=30)
print("test_response",test_response)
test_dict=test_response.json()
print(test_dict)