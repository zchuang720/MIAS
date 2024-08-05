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

import model.object_detection.post as post


""" 调用思路: 模型权重常驻, 每次调用实例化一个作业类ObjectDetector """

class ObjectDetector:

    def __init__(self, postHelper, controller):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.postHelper = postHelper
        self.request = self.postHelper.request
        self.analysisType = self.request.analysisType
        self.controller = controller
        if self.request.modelType == "2":
            self.model = YOLO(os.path.join(self.current_dir, "model", "road_sign.pt"))
        elif self.request.modelType == "4":
            self.model = YOLO(os.path.join(self.current_dir, "model", "soldier_queue.pt"))
        else:
            self.model = YOLO(os.path.join(self.current_dir, "model", "yolov8n.pt"))
        
        self.set_source()

        # 如果分析视频文件， 创建writer
        if self.analysisType == main_poster.analysisType["videoFile"]:
            self.save_path = os.path.join(self.current_dir, "out", "detect-"+str(time.time())+".mp4")
            print('output:\t' + str(self.save_path))
            self.writer = cv2.VideoWriter(self.save_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.shape)
        
        self.interval = int(max(1, self.fps * 1.0))

        # 推理强制启动
        self.model(os.path.join(self.current_dir, "source", "1.jpeg"))

        print("Initialized Boundary-Detector and Tracker")

    def set_source(self):
        if self.analysisType == main_poster.analysisType["stream"]:     # 视频流
            print("Received stream")
            self.source = self.request.url
        elif self.analysisType == main_poster.analysisType["videoFile"]:
            print("Received video file")
            self.source = util.download(url=self.request.file, save_folder= "./tmp/")
        elif self.analysisType != main_poster.analysisType["image"]:
            print("Received image")
            self.source = util.download(url=self.request.file, save_folder= "./tmp/")
            return

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
            result = self.model.predict(frame)[0]   # [0] for the first image
            self.generate_common_post_results(result)
            self.postHelper.post(result=results, image=frame_plot)
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
                result = self.model.predict(frame)[0]   # [0] for the first image
            
            # 绘画检测结果
            frame_plot = copy.deepcopy(frame)
            result.orig_img = frame_plot
            frame_plot = result.plot()
            
            # 发送post
            if self.frame_cnt % self.interval == 0:
                results = self.generate_common_post_results(result)
                if results:
                    if self.analysisType == main_poster.analysisType["videoFile"]:  # 添加视频时间
                        self.postHelper.post(result=results, image=frame_plot, time=self.frame_cnt/self.fps)
                    else:
                        self.postHelper.post(result=results, image=frame_plot)

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
            
        print("Stopped Object Detector")

        # 释放自身对象
        del self
            
    def generate_common_post_results(self, detect_result):
        results = []
        boxes = detect_result.boxes.xywh.cpu().numpy().astype(np.int32)
        for i in range(len(boxes)):
            if self.request.modelType == "2":
                result = copy.deepcopy(post.road_sign_result_dict)
                result["class"] = post.road_sign_names[detect_result.names[int(detect_result.boxes[i].cls.item())]]
            elif self.request.modelType == "4":
                result = copy.deepcopy(post.soldier_queue_result_dict)
                result["class"] = detect_result.names[int(detect_result.boxes[i].cls.item())]

            result["position"] = [int(j) for j in boxes[i]]
            results.append(result)
        return results
    

if __name__ == '__main__':
    startModelRequest = StartModelRequest(analysisType="1", modelType="1", businessId="2")
    modelPostHelper = ModelPostHelper(startModelRequest)
    controller = Controller()

    detector = ObjectDetector(postHelper=modelPostHelper, controller=controller)
    # detector.run()

    detect_thread = threading.Thread(target=detector.run)
    detect_thread.start()
    time.sleep(10)
    controller.stop()
    time.sleep(1)
    print("Main thread exit")

