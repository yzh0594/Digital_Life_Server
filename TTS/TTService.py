import sys
import time

sys.path.append('TTS/vits') # 添加 TTS/vits 模块路径

import soundfile # 用于保存生成的音频文件
import os
os.environ["PYTORCH_JIT"] = "0" # 禁用 PyTorch 的 JIT 特性
import torch

# 导入 VITS 模型相关模块
import TTS.vits.commons as commons
import TTS.vits.utils as utils

from TTS.vits.models import SynthesizerTrn # VITS 模型类
from TTS.vits.text.symbols import symbols # 文本符号
from TTS.vits.text import text_to_sequence # 文本转为序列的工具

import logging
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)


def get_text(text, hps):
    """
    将输入文本转为模型可处理的格式。
    Args:
        text (str): 输入文本。
        hps (Namespace): 模型超参数配置。
    Returns:
        torch.LongTensor: 文本对应的序列张量。
    """
    text_norm = text_to_sequence(text, hps.data.text_cleaners) # 文本标准化并转为序列
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0) # 在序列中插入空白符
    text_norm = torch.LongTensor(text_norm) # 转为 PyTorch 张量
    return text_norm


class TTService():
    def __init__(self, cfg, model, char, speed):       
        """
        初始化 TTS 服务。

        Args:
            cfg (str): 模型配置文件路径。
            model (str): 模型权重文件路径。
            char (str): 当前角色名称。
            speed (float): 语速调整比例。
        """
        logging.info('初始化TTS服务 for %s...' % char)
        self.hps = utils.get_hparams_from_file(cfg) # 加载模型配置
        self.speed = speed # 设置语速
        # 初始化 VITS 模型
        self.net_g = SynthesizerTrn(
            len(symbols), # 文本符号数量 
            self.hps.data.filter_length // 2 + 1, # 频谱滤波器长度的一半加1
            self.hps.train.segment_size // self.hps.data.hop_length, # 训练片段大小
            **self.hps.model # 其他模型参数
            ).cuda()  # 将模型加载到 GPU
        _ = self.net_g.eval() # 设置模型为评估模式
        _ = utils.load_checkpoint(model, self.net_g, None) # 加载模型权重

    def read(self, text):
        """
        将文本合成语音数据。

        Args:
            text (str): 输入文本。

        Returns:
            numpy.ndarray: 生成的音频数据。
        """
        text = text.replace('~', '！')  # 替换特殊字符
        stn_tst = get_text(text, self.hps)  # 将文本转为序列
        with torch.no_grad():  # 禁用梯度计算
            x_tst = stn_tst.cuda().unsqueeze(0)  # 转为 GPU 张量，并增加批次维度
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()  # 获取文本长度
            # 调用模型进行推理，生成音频数据
            audio = self.net_g.infer(
                x_tst, 
                x_tst_lengths, 
                noise_scale=.667,  # 噪声规模
                noise_scale_w=0.2,  # 波形噪声规模
                length_scale=self.speed  # 调整语速
            )[0][0, 0].data.cpu().float().numpy()  # 转为 NumPy 数组
        return audio

    def read_save(self, text, filename, sr):
        """
        将文本合成语音并保存为音频文件。

        Args:
            text (str): 输入文本。
            filename (str): 保存文件的路径。
            sr (int): 采样率。
        """
        stime = time.time()  # 记录开始时间
        au = self.read(text)  # 调用 read 方法生成音频
        soundfile.write(filename, au, sr)  # 将音频保存为文件
        logging.info('VITS语音合成完成, 用时 %.2f' % (time.time() - stime)) 




