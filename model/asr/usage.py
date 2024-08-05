import os
import whisper
import opencc

current_dir = os.path.dirname(os.path.abspath(__file__))


model = whisper.load_model("base")
result = model.transcribe(os.path.join(current_dir, "source", "audio.mp3"))
print(result)
cc = opencc.OpenCC('t2s')
s = cc.convert(result["text"])
print(s)
