import copy
import time
import traceback
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, Form

import util
from main_poster import *
from model.boundary.detect import BoundaryDetector
from model.object_detection.detect import ObjectDetector
from model.car.recognize import CarRecognizer
from model.asr.recogize import SpeechRecognizer, speechRecognizer
from model.asr.TextToSpeech import TTS
# from model.asr.realtime import run_server
from model.face.rocognize import FaceHandler
from model.image_process.process import ImageProcesser
from model.scene.detect import SceneRecognizer

import warnings
warnings.filterwarnings("ignore")


# 全局变量
controller = {}
app = FastAPI()


# 视频分析接口
@app.post('/startModel')
def startModel(analysisType=Form(None), 
                businessId=Form(None),
                url=Form(None),
                file=Form(None),
                modelType=Form(None),
                position=Form(None), 
                modelStopCode=Form(None)
                ):
    response = copy.deepcopy(startModelReturn)

    request = StartModelRequest(analysisType=analysisType,
                                businessId=businessId,
                                url=url,
                                file=file,
                                modelType=modelType,
                                position=position,
                                modelStopCode=modelStopCode)
    if request.position:
        request.position = json.loads(request.position)
        print(request.position)

    modelPostHelper = ModelPostHelper(request)
    # 模型运行控制器
    global controller
    controller[request.modelStopCode] = Controller()

    try:
        if request.modelType == '1':    # 周界检测
            task = BoundaryDetector(modelPostHelper, controller[request.modelStopCode])
        elif request.modelType == '2' or request.modelType == '4':  # 目标检测
            task = ObjectDetector(modelPostHelper, controller[request.modelStopCode])
        elif request.modelType == '3':  # 车辆识别
            task = CarRecognizer(modelPostHelper, controller[request.modelStopCode])
        elif request.modelType == '5':  # 人脸/人物识别
            task = FaceHandler(modelPostHelper, controller[request.modelStopCode])
        elif request.modelType == '6':  # 场景识别
            task = SceneRecognizer(modelPostHelper, controller[request.modelStopCode])
        else:
            raise Exception("Called unimplemented model type.")
            
        task_thread = threading.Thread(target=task.run)
        task_thread.start()

    except Exception as e:
        tb = traceback.format_exc()
        response["message"] = "模型启动失败"
        response["code"] = "1"
        response["tail"] = str(tb)
        print(tb)
    else:
        response["message"] = "模型启动成功"
        response["code"] = "0"
    
    response["timestamp"] = time.time()
    return response

@app.post('/stopModel')
def stopModel(modelStopCode=Form(None)):
    response = copy.deepcopy(stopModelReturn)

    try:
        global controller
        controller[modelStopCode].stop()
        controller.pop(modelStopCode)
    except Exception as e:
        tb = traceback.format_exc()
        response["message"] = "关闭模型失败"
        response["code"] = 1
        response["tail"] = str(tb)
        print(tb)
    else:
        response["message"] = "关闭模型成功"
        response["code"] = 0
    
    response["timestamp"] = time.time()
    return response

# 语音识别接口
@app.post('/speech/recognize')
def speechRecognize(data=Form(None)):
    print(data)
    data = json.loads(data)
    response = copy.deepcopy(start_asr_return)

    try:
        global speechRecognizer
        if not isinstance(speechRecognizer, SpeechRecognizer):
            speechRecognizer = SpeechRecognizer()
        detect_thread = threading.Thread(target=speechRecognizer.run, args=[data])
        detect_thread.start()
    except Exception as e:
        tb = traceback.format_exc()
        response["message"] = "语音识别模型启动失败"
        response["code"] = 1
        response["tail"] = str(tb)
        print(tb)
    else:
        response["message"] = "语音识别模型启动成功"
        response["code"] = 0
    
    response["timestamp"] = time.time()
    return response

# 语音合成接口
@app.post('/speech/systhesis')
def speechRecognize(rate=Form(None),
                    volumeLevel=Form(None),
                    text=Form(None)):
    response = copy.deepcopy(start_asr_return)

    try:
        save_path = './tmp/ss.mp3'
        TTS.run(text, save_path, rate, volumeLevel)
    except Exception as e:
        tb = traceback.format_exc()
        response["message"] = "语音合成模型启动失败"
        response["code"] = 1
        response["tail"] = str(tb)
        print(tb)
    else:
        response["message"] = "语音合成模型启动成功"
        response["code"] = 0
    
    response["timestamp"] = time.time()
    return response

# 图像处理接口
@app.post('/image/process')
def speechRecognize(file=Form(None),
                    modelType=Form(None)):
    
    print("Image-process modelType:", modelType)
    response = copy.deepcopy(image_process_return)
    try:
        response["data"] = ImageProcesser.run(file, modelType)
    except Exception as e:
        tb = traceback.format_exc()
        response["message"] = "图像处理失败"
        response["code"] = 1
        response["tail"] = str(tb)
        print(tb)
    else:
        response["message"] = "图像处理成功"
        response["code"] = 0
    
    print("Finished Image-process modelType:", modelType)
    response["timestamp"] = time.time()
    return response


if __name__ == '__main__':
    uvicorn.run(app=app,
                host="0.0.0.0",
                port=8080,
                workers=1)
    