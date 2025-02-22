import argparse
import os
import socket
import time
import logging
import traceback
from logging.handlers import TimedRotatingFileHandler

import librosa
import soundfile

from utils.FlushingFileHandler import FlushingFileHandler
from ASR import ASRService
from GPT import OllamaService
from TTS import TTService
from SentimentEngine import SentimentEngine

console_logger = logging.getLogger()
console_logger.setLevel(logging.INFO)
FORMAT = '%(asctime)s %(levelname)s %(message)s'
console_handler = console_logger.handlers[0]
console_handler.setFormatter(logging.Formatter(FORMAT))
console_logger.setLevel(logging.INFO)
file_handler = FlushingFileHandler("log.log", formatter=logging.Formatter(FORMAT))
file_handler.setFormatter(logging.Formatter(FORMAT))
file_handler.setLevel(logging.INFO)
console_logger.addHandler(file_handler)
console_logger.addHandler(console_handler)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apiUrl", type=str, nargs='?', required=True)
    parser.add_argument("--model", type=str, nargs='?', required=True)
    parser.add_argument("--stream", type=str2bool, nargs='?', required=True)
    parser.add_argument("--character", type=str, nargs='?', required=True)
    parser.add_argument("--ip", type=str, nargs='?', required=False)
    parser.add_argument("--brainwash", type=str2bool, nargs='?', required=False)
    parser.add_argument("--prompt", type=str, nargs='?', required=True)
    return parser.parse_args()


class Server():
    def __init__(self, args):
        """
        初始化服务器
        """
        # 初始化服务器连接参数
        self.addr = None
        self.conn = None
        logging.info('---Initializing Server---')
        self.host = socket.gethostbyname(socket.gethostname()) # 获取本机IP地址
        self.port = 38438 # 服务器监听的端口
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # 创建TCP/IP套接字
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 10240000)  # 设置发送缓冲区大小
        self.s.bind((self.host, self.port)) # 绑定IP地址和端口
        self.tmp_recv_file = 'tmp/server_received.wav' # 临时接收文件路径
        self.tmp_proc_file = 'tmp/server_processed.wav' # 临时处理后文件路径

        ## 预定义角色映射，每个角色包括模型配置文件路径、权重路径、名称和语速调整比例
        self.char_name = {
            'paimon': ['TTS/models/paimon6k.json', 'TTS/models/paimon6k_390k.pth', 'character_paimon', 1],
            'yunfei': ['TTS/models/yunfeimix2.json', 'TTS/models/yunfeimix2_53k.pth', 'character_yunfei', 1.1],
            'catmaid': ['TTS/models/catmix.json', 'TTS/models/catmix_107k.pth', 'character_catmaid', 1.2]
        }

        # 初始化 ASR 服务
        self.paraformer = ASRService.ASRService('./ASR/resources/config.yaml')
        
        # 初始化 Ollama 服务
        self.ollama = OllamaService.OllamaService(args)

        # 初始化 TTS 服务，基于角色选择加载模型
        self.tts = TTService.TTService(*self.char_name[args.character])

        # 初始化情感分析引擎
        self.sentiment = SentimentEngine.SentimentEngine('SentimentEngine/models/paimon_sentiment.onnx')

    def listen(self):
        """
        服务器主循环：监听客户端连接并处理数据
        """
        while True:
            self.s.listen() # 开始监听连接
            logging.info(f"服务在监听： {self.host}:{self.port}...")
            self.conn, self.addr = self.s.accept()
            logging.info(f"Connected by {self.addr}") # 接收客户端连接
            self.conn.sendall(b'%s' % self.char_name[args.character][2].encode()) # 发送角色名称信息
            while True:
                try:
                    # 接收文件
                    file = self.__receive_file()
                    # print('file received: %s' % file)
                    with open(self.tmp_recv_file, 'wb') as f:
                        f.write(file)
                        logging.info('WAV file received and saved.')
                    # 处理语音文件，转为文本
                    ask_text = self.process_voice()
                    # 根据是否启用流式处理，调用不同的服务
                    if args.stream:
                        for sentence in self.ollama.ask_stream(ask_text):
                            self.send_voice(sentence)
                        self.notice_stream_end()
                        logging.info('Stream finished.')
                    else:
                        resp_text = self.ollama.ask(ask_text)
                        self.send_voice(resp_text)
                        self.notice_stream_end()
                except Exception as e:
                    logging.error(e.__str__())
                    logging.error(traceback.format_exc())
                    break

    def notice_stream_end(self):
        """
        通知客户端流式数据结束
        """
        time.sleep(0.5) # 等待一小段时间，确保数据发送完成
        self.conn.sendall(b'stream_finished')

    def send_voice(self, resp_text, senti_or = None):
        # 调用 TTS 服务将文本生成语音并保存为文件
        self.tts.read_save(resp_text, # 要转换的文本
                           self.tmp_proc_file, # 临时保存生成音频的文件路径
                           self.tts.hps.data.sampling_rate) # 采样率（从 TTS 配置中获取）
        # 打开生成的音频文件并读取内容
        with open(self.tmp_proc_file, 'rb') as f:
            senddata = f.read()
        # 如果传入了情感分析结果 senti_or，就使用它；否则用情感分析模块生成情感分数
        if senti_or:
            senti = senti_or
        else:
            senti = self.sentiment.infer(resp_text)
        # 将情感分数附加到音频数据的末尾
        senddata += b'?!'
        senddata += b'%i' % senti
        # 发送音频和情感数据给客户端
        self.conn.sendall(senddata)
        # 延时（为了防止网络阻塞和适配客户端）
        time.sleep(0.5)
        logging.info('WAV SENT, size %i' % len(senddata))

    def __receive_file(self):
        """
        从客户端接收文件数据，直到接收到结束标志 '?!'
        """
        file_data = b''
        while True:
            data = self.conn.recv(1024) # 每次接收1024字节
            # print(data)
            self.conn.send(b'sb') # 回传数据，表示接收成功
            if data[-2:] == b'?!': # 检查是否到达文件末尾标志
                file_data += data[0:-2]
                break
            if not data:
                # logging.info('Waiting for WAV...')
                continue
            file_data += data

        return file_data

    def fill_size_wav(self):
        """
        填充 WAV 文件的大小信息，确保文件格式正确
        """
        with open(self.tmp_recv_file, "r+b") as f:
            size = os.path.getsize(self.tmp_recv_file) - 8  # 获取文件大小并减去头部偏移
            f.seek(4)
            f.write(size.to_bytes(4, byteorder='little')) # 写入总大小
            f.seek(40)
            f.write((size - 28).to_bytes(4, byteorder='little')) # 写入音频数据大小
            f.flush()

    def process_voice(self):
        """
        处理接收到的语音文件：转为单声道并重采样，然后调用 ASR 转文本
        """
        # stereo to mono
        self.fill_size_wav() # 修正 WAV 文件格式
        y, sr = librosa.load(self.tmp_recv_file, sr=None, mono=False) # 加载音频
        y_mono = librosa.to_mono(y) # 转为单声道
        y_mono = librosa.resample(y_mono, orig_sr=sr, target_sr=16000) # 重采样到16kHz
        soundfile.write(self.tmp_recv_file, y_mono, 16000) # 保存处理后的音频
        text = self.paraformer.infer(self.tmp_recv_file) # 使用 ASR 转为文本
        return text
                

if __name__ == '__main__':
    try:
        args = parse_args() # 解析命令行参数
        s = Server(args) # 初始化服务器
        s.listen() # 开始监听
    except Exception as e:
        logging.error(e.__str__())
        logging.error(traceback.format_exc())
        raise e
