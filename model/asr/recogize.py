import os
import sys
import time
import copy
import requests
import traceback
import whisper
import opencc

sys.path.append(".")
import util
# import post
import main_poster
import model.asr.post as post

class SpeechRecognizer:
    def __init__(self, controller=None):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.controller = controller
        self.model = whisper.load_model("base")
        self.cc = opencc.OpenCC('t2s')
        self.download_path = "./tmp"

    def run(self, source):
        if isinstance(source, dict):
            for audioId in source:
                data = copy.deepcopy(post.result_dict)
                try:
                    print("Detecting audio:", audioId)
                    url = source[audioId]
                    audio_file = util.download(url, self.download_path)
                    
                    start_time = time.time()
                    result = self.model.transcribe(audio_file)
                    end_time = time.time()
                    print("used time:", end_time - start_time)
                    # 繁体中文转简体
                    if result["language"] == "zh":
                        result["text"] = self.cc.convert(result["text"])
                    # 删除下载文件
                    # os.remove(audio_file)
                except Exception as e:
                    tb = traceback.format_exc()
                    data["code"] = "1"
                    print(tb)
                else:
                    data["status"] = "0"
                    data["language"] = result["language"]
                    data["result"] = result["text"]
                    data["audioId"] = audioId

                print(data)
                util.post(main_poster.asr_post_url, data)

        else:
            raise Exception("Unsolved audio sources.")


speechRecognizer = SpeechRecognizer()

if __name__ == '__main__':
    audios = {
        "10001": "http://192.168.65.117:8070/GXZC_IV/business/audioAnalysis/downloadAudio?fileId=61ff98c9-4d70-497e-a7e9-c3e3c881712e",
    }
    speechRecognizer.run(audios)
