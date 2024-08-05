import os
import sys
import cv2
import copy
import threading
import time
import numpy as np
from ultralytics import YOLO

sys.path.append(".")
import main_poster
import util
from main_poster import StartModelRequest, ModelPostHelper, Controller

import model.scene.post as post


""" 调用思路: 模型权重常驻, 每次调用实例化一个作业类ObjectDetector """

class SceneRecognizer:

    def __init__(self, postHelper, controller):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.postHelper = postHelper
        self.request = self.postHelper.request
        self.analysisType = self.request.analysisType
        self.controller = controller
        self.model = YOLO(os.path.join(self.current_dir, "model", "scene.pt"))
        self.id_cls = {}
        
        self.set_source()

        # 如果分析视频文件， 创建writer
        if self.analysisType == main_poster.analysisType["videoFile"]:
            self.save_path = os.path.join(self.current_dir, "out", "detect-"+str(time.time())+".mp4")
            print('output:\t' + str(self.save_path))
            self.writer = cv2.VideoWriter(self.save_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.shape)
        
        self.interval = int(max(1, self.fps * 1.0))

        print("Initialized Scene-Recognizer")

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
            self.source = os.path.join(self.current_dir, "source", "hat.mp4")

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
                result = self.model.track(frame)[0]   # [0] for the first image
                post_results = self.generate_track_post_results(result)
                if post_results:
                    summary = ""
                    for post in post_results:
                        summary += f"id {post['id']}: {post['desc']} {post['class']}\n"

            # 绘画检测结果
            frame_plot = copy.deepcopy(frame)
            result.orig_img = frame_plot
            frame_plot = result.plot()
            for i, text in enumerate(summary.split('\n')):
                cv2.putText(frame_plot, text, (30, 30*(i+1)), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            
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
            
        print("Stopped Scene-Recognizer")

        # 释放自身对象
        del self
    
    def generate_track_post_results(self, curr_result):
        post_results = []
        if curr_result.boxes.id is None or len(curr_result.boxes.id) == 0:
            return post_results
        
        curr_id_cls = {}
        for i in range(len(curr_result.boxes)):
            if curr_result.boxes.id[i]:
                curr_id_cls[int(curr_result.boxes.id[i])] = int(curr_result.boxes.cls[i])

        for id in curr_id_cls:
            post_result = copy.deepcopy(post.scene_result_dict)
            if id not in self.id_cls:
                post_result['desc'] = "new"
            elif id in self.id_cls:
                if curr_id_cls[id] != self.id_cls[id]:
                    post_result['desc'] = "change"
            
            post_result['class'] = post.names_dict[curr_result.names[curr_id_cls[id]]]
            post_result['id'] = id
            # post_result['position'] = [int(j) for j in curr_result.boxes.xywh[i]]

            if post_result['desc']:
                post_results.append(post_result)

        self.id_cls = curr_id_cls
        return post_results





if __name__ == '__main__':
    startModelRequest = StartModelRequest(analysisType="1", modelType="1", businessId="2")
    modelPostHelper = ModelPostHelper(startModelRequest)
    controller = Controller()

    detector = SceneRecognizer(postHelper=modelPostHelper, controller=controller)
    # detector.run()

    detect_thread = threading.Thread(target=detector.run)
    detect_thread.start()
    time.sleep(10)
    controller.stop()
    time.sleep(1)
    print("Main thread exit")

