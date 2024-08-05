import os
import sys
import time
import copy
import requests
import traceback
import cv2
import base64
import json
import easyocr

sys.path.append(".")
import util
# import post
import model.image_process.post as post
import model.image_process.models as image_models
from model.face.rocognize import FaceHandler
from model.car.recognize import CarRecognizer


modelType_output_base64_img = [str(i) for i in range(5,9)]
modelType_output_ocr = [str(i) for i in [2, 4]]

class ImageProcesser:

    @staticmethod
    def run(file_url, modelType):
        image_file = util.download(file_url, "./tmp")

        if modelType == '1':
            output = FaceHandler.recognize_on_image(image_file)
        elif modelType == '2' or modelType == '4':
            reader = easyocr.Reader(['ch_sim','en'], gpu=True)
            output = reader.readtext(image_file)
        elif modelType == '3':
            recognizer = CarRecognizer()
            output = recognizer.recognize_lp_on_image(image_file)
        elif modelType == '5':
            output = image_models.Denoiser.process(image_file)
        elif modelType == '6':
            output = image_models.Dehazer.process(image_file)
        elif modelType == '7':
            output = image_models.Inpainter.process(image_file)
        elif modelType == '8':
            output = image_models.SuperRestorater.process(image_file)

        if modelType in modelType_output_base64_img:
            _, output = cv2.imencode('.jpg', output)
            output = base64.b64encode(output).decode('utf-8')
        elif modelType in modelType_output_ocr:
            # ocr 识别结果数据类型转换
            output_conv = []
            for i in range(len(output)):
                box, text, prob = output[i]
                box = [[int(i[0]), int(i[1])] for i in box]
                prob = float(prob)
                output_conv.append({"box": box, "text": text, "prob": prob})
            output = json.dumps(output_conv)

        return output


