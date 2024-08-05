import sys
import os
sys.path.append('.')
sys.path.append('./model/face')

import logging
import logging.config

import yaml
import json
import copy
import time
import cv2
import torch
import numpy as np
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler

import main_poster
import util
from main_poster import StartModelRequest, ModelPostHelper, Controller
from model.car.LicensePlateOCR import LicensePlateOCR
import model.face.post as post


current_dir = None
faceDetModelHandler = None
faceAlignModelHandler = None
faceRecModelHandler = None
face_cropper = None
face_database_path = None
face_database = {}

def init():
    global current_dir, faceDetModelHandler, faceAlignModelHandler, faceRecModelHandler, face_cropper, \
            face_database_path, face_database
    current_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(current_dir, 'config/model_conf.yaml')) as f:
        model_conf = yaml.load(f, Loader=yaml.FullLoader)

    # common setting for all models, need not modify.
    model_path = os.path.join(current_dir, 'models')

    # face detection model setting.
    scene = 'non-mask'
    model_category = 'face_detection'
    model_name =  model_conf[scene][model_category]
    print('Start to load the face detection model...')
    try:
        faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
        model, cfg = faceDetModelLoader.load_model()
        faceDetModelHandler = FaceDetModelHandler(model, 'cuda:0', cfg)
    except Exception as e:
        print('Falied to load face detection Model.')
        print(e)

    # face landmark model setting.
    model_category = 'face_alignment'
    model_name =  model_conf[scene][model_category]
    print('Start to load the face landmark model...')
    try:
        faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
        model, cfg = faceAlignModelLoader.load_model()
        faceAlignModelHandler = FaceAlignModelHandler(model, 'cuda:0', cfg)
    except Exception as e:
        print('Failed to load face landmark model.')
        print(e)

    # face recognition model setting.
    model_category = 'face_recognition'
    model_name =  model_conf[scene][model_category]    
    print('Start to load the face recognition model...')
    try:
        faceRecModelLoader = FaceRecModelLoader(model_path, model_category, model_name)
        model, cfg = faceRecModelLoader.load_model()
        faceRecModelHandler = FaceRecModelHandler(model, 'cuda:0', cfg)
    except Exception as e:
        print('Failed to load face recognition model.')
        print(e)

    # face utils
    face_cropper = FaceRecImageCropper()

    face_database_path = os.path.join(current_dir, "database")
    try:
        face_database['features'] = np.load(os.path.join(face_database_path, "features.npy"))
        with open(os.path.join(face_database_path, "names")) as file:
            face_database['names'] = json.load(file)
            file.close()
    except Exception as e:
        print(e)
        if not face_database or not face_database['features'] or not face_database['names']:
            face_database['features'] = np.array([])
            face_database['names'] = { '-1': 'Unknown' }

init()


class FaceHandler:
    def __init__(self, postHelper, controller):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.postHelper = postHelper
        self.request = self.postHelper.request
        self.analysisType = self.request.analysisType
        self.controller = controller
        
        self.set_source()

        # 如果分析视频文件， 创建writer
        if self.analysisType == main_poster.analysisType["videoFile"]:
            self.save_path = os.path.join(self.current_dir, "out", "face-"+str(time.time())+".mp4")
            print('output:\t' + str(self.save_path))
            self.writer = cv2.VideoWriter(self.save_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.shape)
        
        self.interval = int(max(1, self.fps * 1.))

        print("Initialized Face-Recognizer")

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
            self.source = os.path.join(self.current_dir, "source", "test1.mp4")

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
                face_boxes, ids, names, probs = FaceHandler.recognize(frame)
            
            # 绘画检测结果
            frame_plot = copy.deepcopy(frame)
            for i in range(len(face_boxes)):
                cv2.rectangle(frame_plot, (face_boxes[i][0], face_boxes[i][1]), (face_boxes[i][2], face_boxes[i][3]), (0, 255, 0), 2)
                cv2.putText(frame_plot, 
                            '%s' % (ids[i]), 
                            (face_boxes[i][0], face_boxes[i][1]), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)     # 打上文字信息
            
            # 发送post
            if self.frame_cnt % self.interval == 0:
                post_results = FaceHandler.generate_post_results(face_boxes, ids, names, probs)
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
    
    @staticmethod
    def generate_post_results(face_boxes, ids, names, probs):
        post_results = []
        # 添加识别结果
        for i in range(len(face_boxes)):
            post_result = copy.deepcopy(post.face_result_dict)
            xywh = [(face_boxes[i][0] + face_boxes[i][2]) / 2,
                    (face_boxes[i][1] + face_boxes[i][3]) / 2,
                    face_boxes[i][2] - face_boxes[i][0],
                    face_boxes[i][3] - face_boxes[i][1]]
            post_result["position"] = [int(j) for j in xywh]
            post_result["id"] = ids[i]
            post_result["name"] = names[i]
            post_result["prob"] = float(probs[i])
            if post_result["prob"] > 0.3:
                post_results.append(post_result)

        return post_results

    @staticmethod
    def recognize(image):
        dets = faceDetModelHandler.inference_on_image(image)
        face_nums = dets.shape[0]

        feature_list = []
        face_boxes = []   # xyxy
        for i in range(face_nums):
            landmarks = faceAlignModelHandler.inference_on_image(image, dets[i])
            landmarks_list = []
            for (x, y) in landmarks.astype(np.int32):
                landmarks_list.extend((x, y))
            cropped_image = face_cropper.crop_image_by_mat(image, landmarks_list)
            feature = faceRecModelHandler.inference_on_image(cropped_image)
            face_boxes.append(dets[i][0:5])
            feature_list.append(feature)

        feature_list = np.array(feature_list)
        face_boxes = np.array(face_boxes).astype(np.int32)
        similarities = np.dot(feature_list, face_database['features'].T)
        
        probs = np.max(similarities, axis=1)
        ids = np.argmax(similarities, axis=1).tolist()

        for i in range(face_nums):
            if probs[i] < 0.1:
                ids[i] = -1
        names = [face_database['names'][str(i)] for i in ids]
            
        print(names)
        print(similarities)

        return face_boxes, ids, names, probs

    @staticmethod
    def register(image, name):
        image = cv2.imread(image)
        dets = faceDetModelHandler.inference_on_image(image)
        face_nums = dets.shape[0]
        if face_nums != 1:
            print("Input image should contain and only contain one face to register!")
            print(f"Detected {face_nums} face.")
            
        landmarks = faceAlignModelHandler.inference_on_image(image, dets[0])
        landmarks_list = []
        for (x, y) in landmarks.astype(np.int32):
            landmarks_list.extend((x, y))
        cropped_image = face_cropper.crop_image_by_mat(image, landmarks_list)
        feature = faceRecModelHandler.inference_on_image(cropped_image)

        if face_database['features'].shape[0] == 0:
            face_database['features'] = np.array([feature, ])
        else:
            face_database['features'] = np.r_[face_database['features'], [feature]]
        register_id = str(len(face_database['names']) - 1)
        face_database['names'][register_id] = name

        np.save(os.path.join(face_database_path, "features"), face_database['features'])
        with open(os.path.join(face_database_path, "names"), 'w') as file:
            json.dump(face_database['names'], file)

        print("Registered face succeed: %s-%s" % (register_id, name))

    @staticmethod
    def recognize_on_image(image):
        image = cv2.imread(image)
        face_boxes, ids, names, probs = FaceHandler.recognize(image)
        post_results = FaceHandler.generate_post_results(face_boxes, ids, names, probs)
        return post_results


if __name__ == "__main__":
    FaceHandler.register("./model/face/api_usage/test_images/person1.jpg", "Musk")
    FaceHandler.register("./model/face/api_usage/test_images/person2.jpg", "Macron")
    # FaceHandler.register("./model/face/api_usage/test_images/person3.jpg", "王佳瑞")
    # FaceHandler.register("./model/face/api_usage/test_images/person4.jpg", "黄治城")
    # FaceHandler.register("./model/face/api_usage/test_images/person5.jpg", "董学尧")
    # FaceHandler.recognize("./model/face/api_usage/test_images/person1_test1.jpg")
    # FaceHandler.recognize("./model/face/api_usage/test_images/person1_test2.jpg")
    # FaceHandler.recognize("./model/face/api_usage/test_images/person1_test3.jpg")
    # FaceHandler.recognize("./model/face/api_usage/test_images/person2_test1.jpg")
    # FaceHandler.recognize("./model/face/api_usage/test_images/person2_test2.jpg")
    # FaceHandler.recognize("./model/face/api_usage/test_images/person2_test3.jpg")
    # FaceHandler.recognize("./model/face/api_usage/test_images/person3_test1.jpg")
    # FaceHandler.recognize("./model/face/api_usage/test_images/person3_test2.jpg")
    # FaceHandler.recognize("./model/face/api_usage/test_images/person3_test3.jpg")
    # FaceHandler.recognize("./model/face/api_usage/test_images/person3_test4.jpg")
    # FaceHandler.recognize("./model/face/api_usage/test_images/person3_test5.jpg")
    # FaceHandler.recognize("./model/face/api_usage/test_images/person4_test1.jpg")
    # FaceHandler.recognize("./model/face/api_usage/test_images/person4_test2.jpg")
    # FaceHandler.recognize("./model/face/api_usage/test_images/person5_test1.jpg")
    # FaceHandler.recognize("./model/face/api_usage/test_images/person5_test2.jpg")
    # FaceHandler.recognize("./model/face/api_usage/test_images/person5_test3.jpg")
    # FaceHandler.recognize("./model/face/api_usage/test_images/person5_test4.jpg")
    # FaceHandler.recognize("./model/face/api_usage/test_images/test2.jpg")
    
    # capture = cv2.VideoCapture("./model/face/api_usage/test_images/test1.mp4")
    
    # while True:

    #     ret, frame = capture.read()
    #     if not ret:
    #         print("Process end")
    #         break

    #     FaceHandler.recognize(frame)

