import os
import sys
import cv2
import copy
import threading
import time
import numpy as np
from ultralytics import YOLO

sys.path.append(".")
sys.path.append("../../")
import main_poster
import util
from main_poster import StartModelRequest, ModelPostHelper, Controller
from model.car.LicensePlateOCR import LicensePlateOCR
import model.car.post as post


class CarRecognizer:

    def __init__(self, postHelper=None, controller=None):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.postHelper = postHelper
        self.controller = controller

        self.car_detect_model = YOLO(os.path.join(self.current_dir, "model", "yolov8n.pt"))
        self.lp_detect_model = YOLO(os.path.join(self.current_dir, "model", "ccpd_fn.pt"))
        self.recognize_model = LicensePlateOCR
        
        if self.postHelper is not None:
            self.request = self.postHelper.request
            self.analysisType = self.request.analysisType

            self.set_source()

            # 如果分析视频文件，创建writer
            if self.analysisType == main_poster.analysisType["videoFile"]:
                self.save_path = os.path.join(self.current_dir, "out", "car-"+str(time.time())+".mp4")
                print('output:\t' + str(self.save_path))
                self.writer = cv2.VideoWriter(self.save_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.shape)
        
            self.interval = int(max(1, self.fps * 1.))

        print("Initialized Car-Recognizer")

    def set_source(self):
        self.source = ""
        if self.analysisType == main_poster.analysisType["stream"]:     # 视频流
            print("Received stream")
            if self.request.url: self.source = self.request.url
        elif self.analysisType == main_poster.analysisType["videoFile"]:
            print("Received video file")
            if self.request.file: self.source = util.download(url=self.request.file, save_folder= "./tmp/")
        elif self.analysisType != main_poster.analysisType["image"]:
            print("Received image")
            self.source = util.download(url=self.request.file, save_folder= "./tmp/")
            return

        if not self.source:
            self.source = os.path.join(self.current_dir, "source", "car.mp4")

        self.capture = cv2.VideoCapture(self.source)
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        self.shape = (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frame_num = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print('source:\t' + str(self.source))
        print('shape:\t' + str(self.shape))
        print('fps:\t' + str(self.fps))
        print('frame:\t' + str(self.frame_num))

        self.display_scale = 0.5
        self.display_shape = (int(self.shape[0] * self.display_scale), int(self.shape[1] * self.display_scale))
        
    def run(self):
        # 图片识别
        if self.analysisType == main_poster.analysisType["image"]:
            car_detect_result, lp_detect_result, post_results = self.pipeline(frame)
            frame_plot = copy.deepcopy(frame)
            frame_plot = car_detect_result.plot()
            lp_detect_result.orig_img = frame_plot
            frame_plot = lp_detect_result.plot()
            # 显示车牌
            for post_result in post_results:
                if post_result["cardNumber"]:
                    cv2.putText(frame_plot, post_result["cardNumber"], 
                                (post_result["position"][0], post_result["position"][1]), 
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)     # 打上文字信息
            self.postHelper.post(result=post_results, image=frame_plot)
            return
        
        # 视频识别流程:
        summary = ""
        self.frame_cnt = 0
        while True:
            if self.controller.is_stop():
                break

            # 读取一帧
            ret, frame = self.capture.read()
            if not ret:
                print("Process end")
                break

            # 每interval帧检测一次
            if self.frame_cnt % self.interval == 0:
                print("Detected frame #", self.frame_cnt)
                car_detect_result, lp_detect_result, post_results = self.pipeline(frame)
            
            # 绘画检测结果
            frame_plot = copy.deepcopy(frame)
            car_detect_result.orig_img = frame_plot
            frame_plot = car_detect_result.plot()
            lp_detect_result.orig_img = frame_plot
            frame_plot = lp_detect_result.plot()
            # 显示车牌
            for post_result in post_results:
                if post_result["cardNumber"]:
                    cv2.putText(frame_plot, post_result["cardNumber"], 
                                (post_result["position"][0], post_result["position"][1]), 
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)     # 打上文字信息
            
            # 发送post
            if self.frame_cnt % self.interval == 0:
                if post_results:
                    if self.analysisType == main_poster.analysisType["videoFile"]:  # 添加视频时间
                        self.postHelper.post(result=post_results, image=frame_plot, time=self.frame_cnt/self.fps)
                    else:
                        self.postHelper.post(result=post_results, image=frame_plot)

            if hasattr(self, "writer") and self.writer:
                self.writer.write(frame_plot)

            # 可视化
            if not hasattr(self, "writer") or not self.writer:
                frame_plot = cv2.resize(frame_plot, self.display_shape)
                cv2.imshow('pred', frame_plot)
                if cv2.waitKey(int(100 / self.fps)) & 0xFF == ord('q'):
                    break

            self.frame_cnt += 1

        # 写完视频并发送
        if hasattr(self, "writer") and self.writer:
            self.writer.release()
            self.postHelper.post_file(self.save_path, modelType='1')
            
        print("Stopped Model")

        # 释放自身对象
        del self
    
    def pipeline(self, frame):
        # 1 bicycle 2 car 3 motorcycle 4 airplane 5 bus 6 train 7 truck 8 boat
        car_detect_result = self.car_detect_model.track(frame, classes=range(1,9))[0]
        lp_detect_result = self.lp_detect_model.predict(frame)[0]

        car_xyxy = car_detect_result.boxes.xyxy.cpu().numpy().astype(np.int32)
        lp_xyxy = lp_detect_result.boxes.xyxy.cpu().numpy().astype(np.int32)

        lp_texts = []
        for xyxy in lp_xyxy:
            lp_image = copy.deepcopy(frame)
            lp_image = lp_image[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
            lp_texts.append(self.recognize_model.recognize(lp_image))
        
        lp_in_car = self.find_lp_in_which_car(lp_xyxy, car_xyxy)

        post_results = self.generate_post_results(car_detect_result, lp_in_car, lp_texts)

        return car_detect_result, lp_detect_result, post_results

    def find_lp_in_which_car(self, lp_xyxy, car_xyxy):
        lp_centers = np.array([[(lp_xyxy[i][0] + lp_xyxy[i][2]) / 2, (lp_xyxy[i][1] + lp_xyxy[i][3]) / 2] 
                                for i in range(len(lp_xyxy))]).astype(np.int32)
        lp_in_car = []
        for i, center in enumerate(lp_centers):
            for j, car in enumerate(car_xyxy):
                if center[0] >= car[0] and center[0] <= car[2] and \
                    center[1] >= car[1] and center[1] <= car[3]:
                    lp_in_car.append(j)
                    break
        return lp_in_car
        
    def generate_post_results(self, car_detect_result, lp_in_car, lp_texts):
        post_results = []
        car_xywh = car_detect_result.boxes.xywh.cpu().numpy().astype(np.int32)
        # 添加车辆信息
        for i in range(len(car_xywh)):
            result = copy.deepcopy(post.car_result_dict)
            result["position"] = [int(j) for j in car_xywh[i]]
            result["carType"] = post.car_name[car_detect_result.names[int(car_detect_result.boxes[i].cls.item())]]
            post_results.append(result)
        # 添加车牌号
        for i, car_index in enumerate(lp_in_car):
            post_results[car_index]["cardNumber"] = lp_texts[i]

        return post_results
        
    def recognize_on_image(self, image_file):
        image = cv2.imread(image_file)
        car_detect_result, lp_detect_result, post_results = self.pipeline(image)
        return post_results
    
    def recognize_lp_on_image(self, image_file):
        image = cv2.imread(image_file)

        lp_detect_result = self.lp_detect_model.predict(image)[0]
        lp_xyxy = lp_detect_result.boxes.xyxy.cpu().numpy().astype(np.int32)

        lp_texts = []
        if len(lp_xyxy) > 0:
            for xyxy in lp_xyxy:
                lp_image = copy.deepcopy(image)
                lp_image = lp_image[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                lp_texts.append(self.recognize_model.recognize(lp_image))
        else:
            lp_texts.append(self.recognize_model.recognize(image))

        return lp_texts