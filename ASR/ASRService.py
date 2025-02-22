import logging
import time

from ASR.rapid_paraformer import RapidParaformer


class ASRService():
    def __init__(self, config_path):
        """
        初始化 ASR 服务，加载 RapidParaformer 模型。

        Args:
            config_path (str): 模型配置文件的路径。
        """
        logging.info('初始化ASR服务...')
        self.paraformer = RapidParaformer(config_path) # 根据配置文件初始化 ASR 模型实例

    def infer(self, wav_path):
        """
        使用 ASR 模型对音频文件进行推理，返回识别结果。

        Args:
            wav_path (str): 待识别的音频文件路径。

        Returns:
            str: 识别出的文本结果。
        """
        stime = time.time()
        result = self.paraformer(wav_path)
        logging.info('ASR 识别结果: %s. 用时 %.2f.' % (result, time.time() - stime))
        return result[0]

if __name__ == '__main__':
    config_path = 'ASR/resources/config.yaml' # ASR 模型的配置文件路径

    service = ASRService(config_path) # 初始化 ASR 服务

    wav_path = 'ASR/test_wavs/0478_00017.wav' # 待测试的音频文件路径
    result = service.infer(wav_path) # 进行语音识别
    print(result) # 输出识别结果