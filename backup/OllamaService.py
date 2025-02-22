import logging
import os
import time
import requests
import json
import GPT.tune as tune
                  
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
        

    def ask(self, text):
        stime = time.time()
        # 构造请求数据
        payload = {
                  "model": self.model,
                  "messages": [
                      self.system_message,
                      {"role": "user", "content": text},
                  ],
                  "stream": False
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
        complete_text = ""
        stime = time.time()
        
        # 构造请求数据
        payload = {
                  "model": self.model,
                  "messages": [
                      self.system_message,
                      {"role": "user", "content": text},
                  ],
        }
        logging.info(f"发送请求至 {self.api_url}: {payload}")
        
        response = requests.post(self.api_url, json=payload, stream=True)  # 开启流式请求

        if response.status_code == 200:
            # 按行解析流式响应
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        # 解析每行数据
                        data = json.loads(line)
                        response_text = data.get("message", {}).get("content", "")
                        done = data.get("done", False)

                        # 拼接返回的部分
                        complete_text += response_text

                        # 当文本包含标点符号（如句号、感叹号、问号等），且累积文本长度大于一定阈值时，实时输出并发送给 TTS 服务
                        if ("。" in response_text or "！" in response_text or "？" in response_text or "\n" in response_text) and len(complete_text) > 3:
                            logging.info(f'Ollama Stream Response: {complete_text.strip()}, @Time: {time.time() - stime:.2f} seconds')
                            yield complete_text.strip()  # 返回流式文本
                            complete_text = ""  # 重置累积文本

                        # 如果完成，退出流式处理
                        if done:
                            logging.info(f'Ollama Stream Completed: {complete_text.strip()}, @Time: {time.time() - stime:.2f} seconds')
                            break

                    except json.JSONDecodeError:
                        logging.error("Failed to decode JSON response.")
                        continue

            # 如果还有未输出的文本，最终输出
            if complete_text.strip():
                logging.info(f'Ollama Stream Final Response: {complete_text.strip()}, @Time: {time.time() - stime:.2f} seconds')
                yield complete_text.strip()
        else:
            logging.error(f"Request failed with status code {response.status_code}")
            
