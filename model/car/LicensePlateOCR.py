import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import sys
import time
import argparse

sys.path.append(".")
sys.path.append("../../")
from model.car.ocr.plateNet import myNet_ocr
from model.car.ocr.alphabets import plate_chr
from model.car.ocr.LPRNet import build_lprnet


mean_value,std_value=(0.588,0.193)

def decodePlate(preds):
    pre=0
    newPreds=[]
    for i in range(len(preds)):
        if preds[i]!=0 and preds[i]!=pre:
            newPreds.append(preds[i])
        pre=preds[i]
    return newPreds

def image_processing(img,device,img_size):
    img_h,img_w= img_size
    img = cv2.resize(img, (img_w,img_h))
    # img = np.reshape(img, (48, 168, 3))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - mean_value) / std_value
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    return img

def get_plate_result(img,device,model,img_size):
    # img = cv2.imread(image_path)
    input = image_processing(img,device,img_size)
    preds = model(input)
    preds =preds.argmax(dim=2)
    # print(preds)
    preds=preds.view(-1).detach().cpu().numpy()
    newPreds=decodePlate(preds)
    plate=""
    for i in newPreds:
        plate += plate_chr[int(i)]
    return plate

def init_model(device,model_path):
    check_point = torch.load(model_path,map_location=device)
    model_state = check_point['state_dict']
    cfg = check_point['cfg']
    model = myNet_ocr(num_classes=len(plate_chr), export=True, cfg=cfg)        #export  True 用来推理
    # model =build_lprnet(num_classes=len(plate_chr),export=True)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model

def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)


current_dir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=os.path.join(current_dir, 'ocr', 'model', 'best.pth'), help='model.pt path(s)')  
parser.add_argument('--image_path', type=str, default=os.path.join(current_dir, 'ocr', 'source'), help='source')
parser.add_argument('--img_h', type=int, default=48, help='height') 
parser.add_argument('--img_w', type=int, default=168, help='width')
parser.add_argument('--LPRNet', action='store_true', help='use LPRNet')         #True代表使用LPRNet ,False代表用plateNet
parser.add_argument('--acc', type=bool, default=False, help='get accuracy')     #标记好的图片，计算准确率
opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = (opt.img_h, opt.img_w)
model = init_model(device, opt.model_path)


class LicensePlateOCR:

    @staticmethod
    def recognize(image):
        # 判断转换颜色空间
        if image.shape[-1] != 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        result = get_plate_result(image, device, model, img_size)
        return result


if __name__ == "__main__":
    file_list=[]
    allFilePath(opt.image_path,file_list)
    for pic_ in file_list:
        try:
            pic_name = os.path.basename(pic_)
            img = cv2.imread(pic_)
            result = LicensePlateOCR.recognize(img)
            print(result, pic_name)
        except Exception as e:
            print(e)
