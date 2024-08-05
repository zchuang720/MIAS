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

import model.boundary.post as post
from model.boundary.track import BoundaryTracker
# import poster as poster
# from tracking import BoundaryTracker

sources = ["1306标_文华路站-动火场景.mp4",
           "bus.jpg",
           "http://121.89.174.221:8181/live/monitoring/1306.flv",]

""" 调用思路: 模型权重常驻, 每次调用实例化一个作业类BoundaryDetector """

class BoundaryDetector:

    def __init__(self, postHelper, controller):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.postHelper = postHelper
        self.request = self.postHelper.request
        self.analysisType = self.request.analysisType
        self.controller = controller
        self.model = YOLO(os.path.join(self.current_dir, "model", "yolov8n.pt"))
        
        self.set_source()

        # 如果分析视频文件， 创建writer
        if self.analysisType == main_poster.analysisType["videoFile"]:
            self.save_path = os.path.join(self.current_dir, "out", "boundary-"+str(time.time())+".mp4")
            print('output:\t' + str(self.save_path))
            self.writer = cv2.VideoWriter(self.save_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.shape)
        
        self.interval = int(max(1, self.fps * 1.0))
        self.tracker = BoundaryTracker(self.contours)

        # 推理强制启动
        self.model(os.path.join(self.current_dir, "source", sources[1]))

        print("Initialized Boundary-Detector and Tracker")

    def set_source(self):
        if self.analysisType == main_poster.analysisType["stream"]:     # 视频流
            print("Received stream")
            self.source = self.request.url
        elif self.analysisType == main_poster.analysisType["videoFile"]:
            print("Received file.")
            self.source = util.download(url=self.request.file, save_folder= "./tmp/")

        if not self.source:
            self.source = os.path.join(self.current_dir, "source", sources[0])

        self.capture = cv2.VideoCapture(self.source)
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        self.shape = (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frame_num = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # self.capture.set(cv2.CAP_PROP_POS_FRAMES, 1300)
        
        print('source:\t' + str(self.source))
        print('shape:\t' + str(self.shape))
        print('fps:\t' + str(self.fps))
        print('frame:\t' + str(self.frame_num))

        self.contours = self.request.position
        if not self.contours:
            self.contours = [[0.1, 0.4], [0.5, 0.2], [0.9, 0.4], [0.3, 0.8]]
        # 相对坐标 => 像素坐标
        self.contours = np.array([[int(i[0] * self.shape[0]), int(i[1] * self.shape[1])] for i in self.contours])

        self.display_scale = 0.5
        self.display_shape = (int(self.shape[0] * self.display_scale), int(self.shape[1] * self.display_scale))
        
        if hasattr(self, 'tracker'):
            self.tracker.reset()
        
    def run(self):
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
                print("Detect frame #", self.frame_cnt)
                result = self.model.track(frame, imgsz=1280, classes=0)[0]  # [0] for the first image

                summary = self.tracker.track(result)
                dists = self.tracker.dists

                # 获取box和中心点
                boxes = result.boxes.xyxy.cpu().numpy().astype(np.int32)
                centers = np.array([[(boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2] 
                                        for i in range(len(boxes))]).astype(np.int32)
            
            # 绘画检测结果
            frame_plot = copy.deepcopy(frame)
            cv2.polylines(frame_plot, [self.contours], True, (255,0,0), 3)   # 画周界线
            result.orig_img = frame_plot
            frame_plot = result.plot(probs=False)      # 绘制目标框
            cv2.putText(frame_plot, summary, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,255,0), 3)     # 打上文字信息
            for i, center in enumerate(centers):
                if dists[i] > 0:    # 标识界内目标
                    cv2.putText(frame_plot, "INSIDE", (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            
            # 发送post
            if self.frame_cnt % self.interval == 0:
                if_post, results = self.test_boundary(result)
                if if_post:
                    if self.analysisType == main_poster.analysisType["videoFile"]:  # 视频文件转换时间
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
            
        print("Stopped Boundary Detector and Tracker")

        # 释放自身对象
        del self

    def test_boundary(self, result):
        if not self.tracker.result:
            return False, None
        num_enter, num_leave = len(self.tracker.result["enter_boundary"]), len(self.tracker.result["leave_boundary"])
        if num_enter == 0 and num_leave == 0:
            return False, None
        
        results = []
        for id in self.tracker.result["enter_boundary"]:
            type = result.names[self.tracker.id_cls[id]]
            position = [int(i) for i in self.tracker.id_box[id]]
            desc = f"工作人员{int(id)}进入场地"
            results.append(self.generate_one_post_result(desc=desc, type=type, position=position))
        for id in self.tracker.result["leave_boundary"]:
            type = result.names[self.tracker.id_cls[id]]
            position = [int(i) for i in self.tracker.id_box[id]]
            desc = f"工作人员{int(id)}离开场地"
            results.append(self.generate_one_post_result(desc=desc, type=type, position=position))
        return True, results
            
    def generate_one_post_result(self, **kwargs):
        result_dict = copy.deepcopy(post.result_dict)
        result_dict.update(kwargs)
        return result_dict


if __name__ == '__main__':
    startModelRequest = StartModelRequest(analysisType="1", modelType="1", businessId="2")
    modelPostHelper = ModelPostHelper(startModelRequest)
    controller = Controller()

    detector = BoundaryDetector(postHelper=modelPostHelper, controller=controller)
    # detector.run()

    detect_thread = threading.Thread(target=detector.run)
    detect_thread.start()
    time.sleep(10)
    controller.stop()
    time.sleep(1)
    print("Main thread exit")

