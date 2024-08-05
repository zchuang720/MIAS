import requests
import base64
import json
import time
import hashlib
import threading
import copy
import cv2
import pydantic
import typing
import os

# post_url = "http://121.89.174.221:8502/GXZC_IV/api/receiveData"
post_url = "http://192.168.65.117:8070/GXZC_IV/api/receiveData"
push_stream_url = "http://121.89.174.221:8181/live/monitoring/1-result.flv"
upload_vudio_url = "http://192.168.65.117:8070/GXZC_IV/api/uploadVideo"
upload_audio_url = "http://192.168.65.117:8070/GXZC_IV/api/uploadAudio"
asr_post_url = "http://192.168.65.117:8070/GXZC_IV/api/receiveAudio"


analysisType = {
    "stream": "0",
    "videoFile": "1",
    "image": "2",
}

class StartModelRequest():
    """ modelType:
        1、周界入侵（闲杂人员进入周界）
        2、路牌识别
        3、车辆识别
        4、军人阵列
        5、人物识别
        6、未佩戴军帽（场景识别）
    """
    def __init__(self, analysisType=None, businessId=None, url=None, file=None, modelType=None, position=None, modelStopCode=None):
        self.analysisType = analysisType    # 0：摄像头视频流分析 1：视频文件分析 2：图片分析
        self.businessId = businessId        # 业务id（监控号/视频id/音频id）
        self.url = url
        self.file = file
        self.modelType = modelType
        self.position = position            # 多边形坐标 非必填
        self.modelStopCode = modelStopCode  # 停止码

# 返回启动模型请求的结果的数据结构
startModelReturn = {
    "message": None,    # "模型启动成功/失败"
    "code": None,       # 0/1
    "timestamp": None,
    "tail": "",       # 详情或错误信息
}


# 接口二：模型关闭
class StopModelRequest():
    def __init__(self, uuid=None):
        self.uuid = uuid

stopModelReturn = {
    "message": None,    # "模型关闭成功/失败"
    "code": None,       # 0/1
    "timestamp": None,
    "tail": "",       # 详情或错误信息
}

# 交大 to 15s 返回检测结果数据字典
post_dict = {
    "businessId": "",
    "modelType": None,      # 与启动模型请求的modelType一致
    "image": "",            # 事件画面大图base64
    "result": "",           # 检测结果 JSON格式[列表]
    "time": None,
    "analysisType": None,   # 与启动模型请求的一致
    "pullUrl": push_stream_url,
    "extend": "",           # 预留扩展字段 
}

upload_dict = {
    "businessId": "2",      # 业务id
    "modelType": "1"
}

# 语音识别接口
start_asr_return = {
    "message": "",
    "code": None,
    "timestamp": None,
    "tail": ""
}

# 图像处理接口
image_process_return = {
    "message": "",
    "code": None,
    "timestamp": None,
    "data": ""
}

# 交大 to 15s 返回检测结果 辅助POST方法类
class ModelPostHelper:
    def __init__(self, request=None):
        self.post_url = post_url
        self.pull_url = push_stream_url
        self.upload_url = upload_vudio_url
        self.request = request

    def post(self, **kwargs):
        # 创建新post字典
        data = copy.deepcopy(post_dict)
        data["businessId"] = self.request.businessId
        data["modelType"] = self.request.modelType
        data["analysisType"] = self.request.analysisType
        data["time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

        # 更新post内容
        data.update(kwargs)
        if not isinstance(data["result"], str):
            data["result"] = json.dumps(data["result"])
        if not isinstance(data["image"], str):
            _, img_base64 = cv2.imencode('.jpg', cv2.resize(data["image"], (0,0), fx=0.3, fy=0.3))
            img_base64 = base64.b64encode(img_base64).decode('utf-8')
            data["image"] = img_base64
            # print("Converted base64 image")
        
        # print(str(data))

        # 开新线程进行post
        def post_func(url, data, info=True):
            r = requests.post(url, data)
            if info:
                print(r)
                print(r.text)
        
        post_thread = threading.Thread(target=post_func, args=(self.post_url, data, True))
        post_thread.start()

    def post_file(self, file_path, **kwarg):
        # 开新线程进行post
        def post_func(url, data, files, info=True):
            print("Uploading:", str(files["file"]))
            r = requests.post(url, data=data, files=files)
            print("Finished upload")
            if info:
                print(r)
                print(r.text)
            files["file"].close()
            os.remove(file_path)

        data = copy.deepcopy(upload_dict)
        data.update(kwarg)
        data["modelType"] = self.request.modelType
        data["businessId"] = self.request.businessId

        files = {"file": open(file_path, "rb")}

        post_thread = threading.Thread(target=post_func, args=(self.upload_url, data, files, True))
        post_thread.start()
        
# 控制模型运行状态的信号类
class Controller:
    def __init__(self):
        self.signal = False # 停止信号

    def stop(self):
        self.signal = True

    def start(self):
        self.signal = False
    
    def is_stop(self):
        return self.signal
    