import pyttsx3
import time
import util
import main_poster

class TTS:
    @staticmethod
    def run(content=None, save_path='./tmp/ss.mp3', rate=100, volume=0.5):
        start_time = time.time()
        # 创建对象
        engine = pyttsx3.init()

        print(engine.getProperty('voices'))
        # 获取当前语音速率
        # rate = engine.getProperty('rate')
        engine.setProperty('rate', rate)
        print(f'语音速率：{rate}')
        # 设置新的语音速率
        ######################################

        # 获取当前语音音量
        # volume = engine.getProperty('volume')
        # 设置新的语音音量，音量最小为 0，最大为 1
        ###########################################
        engine.setProperty('volume', volume)
        print(f'语音音量：{volume}')
        # 获取当前语音声音的详细信息
        voices = engine.getProperty('voices')

        print(f'语音声音详细信息：{voices}')
        # 设置当前语音声音为女性，当前声音不能读中文
        # engine.setProperty('voice', voices[1].id)
        # 设置当前语音声音为男性，当前声音可以读中文
        engine.setProperty('voice', voices[0].id)
        # 获取当前语音声音
        voice = engine.getProperty('voice')
        print(f'语音声音：{voice}')

        # 语音播报内容
        if not content:
            content = "你好呀，很高兴认识你，我叫王家瑞，我和交通大学的董学尧和黄治城同学正在研发智能分析系统，今天就快要收尾啦，你好呀，很高兴认识你，我叫王家瑞，我和交通大学的董学尧和黄治城同学正在研发智能分析系统，今天就快要收尾啦，你好呀，很高兴认识你，我叫王家瑞，我和交通大学的董学尧和黄治城同学正在研发智能分析系统，今天就快要收尾啦"

        # content = "hello everyone.Nice to meet you here."
        # 输出文件格式

        # # 语音文本
        # path = 'test.txt'
        # with open(path, encoding='utf-8') as f_name:
        #     words = str(f_name.readlines()).replace(r'\n', '')
        engine.save_to_file(content, save_path)
        # engine.say(content)  # 将语音文本说出来
        engine.runAndWait()
        # 将文字输出为 aiff 格式的文件
        engine.stop()
        print(time.time() - start_time)

        data = {
            "key": "value"
        }
        
        util.post_file(main_poster.upload_audio_url, file_path=save_path, data=data)


if __name__ == '__main__':
    TTS.run(rate=100, volume=0.5, save_path='./tmp/ss.mp3')
