import logging
import os
import time
import requests
import json
import GPT.tune as tune
import sys

# 解决 Windows 默认 GBK 不能编码 emoji 的问题
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # 控制台输出
        logging.FileHandler("ollama_service.log", encoding="utf-8"),  # 日志文件输出（UTF-8）
    ]
)
                  
class OllamaService():
    def __init__(self, args ):
        logging.info('初始化Ollama服务...')
        # 初始化 OllamaServer
        self.api_url = args.apiUrl
        
        self.model = args.model
        
        self.tune = tune.get_tune(args.character, args.prompt)
        
        self.system_message = {
             "role": "system",
             "content": (
                  self.tune
             )
         }

        self.counter = 0

        self.brainwash = args.brainwash
        
        self.messages = [self.system_message]  # 存储完整对话历史
        

    def ask(self, text):
        stime = time.time()
        # 构造请求数据
        payload = {
                  "model": self.model,
                  "messages": [
                      self.system_message,
                      {"role": "user", "content": text},
                  ],
                  "stream": True if self.model in ["deepseek-chat", "deepseek-reasoner"] else "true"
        }
        # POST请求与Ollama交互
        response = requests.post(self.api_url, json=payload)
        if response.status_code == 200:
            prev_text = response.json().get("data","")
            logging.info(f'Ollama Response: {prev_text}, time used: {time.time() - stime:.2f} seconds')
            return prev_text
        else:
            logging.error(f'Error with API request: {response.status_code}')
            return "Error occurred while fetching response from Ollama."

    def ask_stream(self, text):
        response_text = ""  # AI的完整回答
        response_chunk = "" # AI的回答块
        stime = time.time()
        
        self.messages.append({"role": "user", "content": text})
        
        # 设置请求头，包含 Bearer Token
        headers = {
            "Authorization": f'Bearer {"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjJmNjQ1NDU5LTE1ZTYtNGQ1ZC04YzE2LWIyMzNkNWNlM2YwYSJ9.GWmrDVWaXGesRxuYhSTCx5Ja7oWwuAQ0v4i63cO4GP0"}',
            "Content-Type": "application/json"
        }
        
        # 构造请求数据
        payload = {
                  "model": self.model,
                  "messages": self.messages,  # 传入整个对话历史
                #   "files": [
                #         {"type": "collection", "id": "e52a1bf5-e783-4c86-a2a8-ab4f0759d1b4"}
                #    ],
                  "stream": True
        }
        
        logging.info(f"发送请求至 {self.api_url}: {payload}")
        
        response = requests.post(self.api_url, json=payload,headers=headers, stream=True)  # 开启流式请求
        
        in_think_block = False  # 是否在 <think> 块内

        if response.status_code == 200:
            # 按行解析流式响应
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        if not line.strip(): 
                            logging.warning("收到空行，可能是服务器端在等待生成")
                            continue
                        # SSE 格式的每个数据块以 'data:' 开头
                        if line.startswith("data:"):
                        # 获取 'data:' 后面的内容并去掉前后的空白字符
                            data_str = line[5:].strip()

                        if not data_str or data_str == "[DONE]":  # 先检查是否为空或者是 [DONE]
                            continue

                        # 尝试解析 JSON 数据
                        data = json.loads(data_str)
                        
                        content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        
                        if content:
                            if "<think>" in content:
                                in_think_block = True  # 进入 "思考模式"
                            if "</think>" in content:
                                in_think_block = False  # 退出 "思考模式"
                                content = content.split("</think>", 1)[1]  # 只保留 `</think>` 后的内容

                            if not in_think_block:  # 仅在不处于 <think> 块时才处理文本
                                response_chunk += content

                        # 当文本包含标点符号（如句号、感叹号、问号等），且累积文本长度大于一定阈值时，实时输出并发送给 TTS 服务
                        if ("，" in content or "。" in content or "！" in content or "？" in content or "\n" in content) and len(response_chunk) > 5 :
                            logging.info(f'Ollama服务响应: {response_chunk.strip()}, @耗时: {time.time() - stime:.2f} 秒')
                            response_text += response_chunk.strip()
                            logging.debug(f"即将输出: {response_chunk.strip()}")
                            yield response_chunk.strip()  # 返回流式文本
                            response_chunk = ""  # 重置累积文本

                    except json.JSONDecodeError:
                        logging.error(f"解析JSON失败: {data_str}")  # 打印错误的数据
                        continue

            # 如果还有未输出的文本，最终输出
            if response_chunk.strip():
                response_text += response_chunk.strip()
                logging.info(f'Ollama Stream Final Response: {response_chunk.strip()}, @Time: {time.time() - stime:.2f} seconds')
                yield response_chunk.strip()
                
            self.messages.append({"role": "assistant", "content": response_text.strip()})
        else:
            logging.error(f"请求失败: {response.status_code} {response.text}")
            
