import requests
import json
import time
import cv2

from ultralytics import YOLO

from util import *
from main_poster import *
from model.car.LicensePlateOCR import LicensePlateOCR

stop_model_request = {
    "modelStopCode": "001"
}


def test1():
    file_url = "http://192.168.65.117:8070/GXZC_IV/business/audioAnalysis/downloadAudio?fileId=61ff98c9-4d70-497e-a7e9-c3e3c881712e"
    download(file_url, "./tmp/")

def test2():
    start_model_request = {
        "businessId": "2",
        "analysisType": analysisType["videoFile"],      # 0：摄像头视频流分析 1：视频文件分析 2：图片分析
        # "url": "http://192.168.65.202:8080/live/monitoring/army.flv",
        # "file": "http://192.168.65.117:8070/GXZC_IV/business/videoAnalysis/getVideoAll?fileId=7d20d846-5d96-4322-ae05-cccbc713a416",
        "url": "",
        "file": "",
        "modelType": '5',
        "postion": "[[0.17, 0.21], [0.17, 0.57], [0.51, 0.57], [0.51, 0.21]]",        # 多边形坐标 非必填
        "modelStopCode": '001'
    }
    response = requests.post("http://127.0.0.1:8080/startModel", data=start_model_request)
    print(response)
    print(response.text)

    time.sleep(20)
    
    response = requests.post("http://127.0.0.1:8080/stopModel", data=stop_model_request)
    print(response)
    print(response.text)

def test3():
    postHelper = ModelPostHelper()
    postHelper.post_file(file_path="./model/boundary/source/bus.jpg",
                         businessId="2",
                         modelType="1")
    
def test4():
    audios = {
        "10001": "http://192.168.65.117:8070/GXZC_IV/business/audioAnalysis/downloadAudio?fileId=61ff98c9-4d70-497e-a7e9-c3e3c881712e",
    }
    response = requests.post("http://127.0.0.1:8080/speech/recognize", data={"data": json.dumps(audios)})
    print(response)
    print(response.text)

def test_GPU_memory():
    yolos = [YOLO("./model/boundary/model/yolov8n.pt"),
              YOLO("./model/boundary/model/yolov8n.pt"),    # 充当车牌
              YOLO("./model/object_detection/model/road_sign.pt"),
              YOLO("./model/object_detection/model/soldier_queue.pt"),
              ]
    ocr = [LicensePlateOCR,]
    for yolo in yolos:
        yolo("./model/boundary/source/bus.jpg")
    for model in ocr:
        print(model.recognize(cv2.imread("./model/boundary/source/bus.jpg")))

    time.sleep(10)

if __name__ == '__main__':
    test2()
    # test_GPU_memory()