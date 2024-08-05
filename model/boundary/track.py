import os
import sys
import cv2
import numpy as np
from ultralytics.engine.results import Results

class BoundaryTracker:
    def __init__(self, contours, info_level=0, direction=2, **kwargs):
        self.contours = contours
        self.info_level = info_level    # 0：打印出入界目标的信息 1：打印出入界和新出现的目标信息
        self.direction = direction  # 0：出界 1：入界 2：出入界
        self.reset()

    def reset(self):
        self.dists = []
        self.id_status = {}     # 0：在界外 1：在界内
        self.id_box = {}        # xywh 格式
        self.id_cls = {}
        self.summary = ""
        self.result = {}
    
    def track(self, result):
        # 获取box和中心点
        box_num = len(result.boxes)
        xyxy = result.boxes.xyxy.cpu().numpy().astype(np.int32)
        centers = np.array([[(xyxy[i][0] + xyxy[i][2]) / 2, (xyxy[i][1] + xyxy[i][3]) / 2] 
                                for i in range(len(xyxy))]).astype(np.int32)
        # 判断box是否在界限内
        self.dists = []
        for center in centers:
            self.dists.append(cv2.pointPolygonTest(self.contours, center.tolist(), measureDist=False))
        # 遇到 result.boxes.id 为空（目标过小等情况）跳过检测
        if result.boxes.id is None:
            print("!!!!!!!!!!! boxes.id is None !!!!!!!!!!!!!!")
            return self.summary
        ids = result.boxes.id.tolist()
        appear_inside = []
        appear_outside = []
        disappear_inside = []
        disappear_outside = []
        enter_boundary = []
        leave_boundary = []
        # 整理新状态
        new_status = {}
        self.id_box.clear()
        self.id_cls.clear()
        for i in range(box_num):
            new_status[ids[i]] = self.dists[i]
            self.id_box[ids[i]] = [xyxy[i][0], xyxy[i][1], xyxy[i][2]-xyxy[i][0], xyxy[i][3]-xyxy[i][1]]
            self.id_cls[ids[i]] = int(result.boxes.cls[i].item())
        
        # 判断新出现的目标
        for id in new_status:
            if id not in self.id_status:
                if new_status[id] > 0:
                    appear_inside.append(id)    # 界内出现
                else:
                    appear_outside.append(id)   # 界外出现

        for id in self.id_status:
            if id not in new_status:    # 判断消失目标
                if self.id_status[id] > 0:
                    disappear_inside.append(id)     # 界内消失
                else:
                    disappear_outside.append(id)    # 界外消失
            else:   # 判断出入界
                if self.id_status[id] >= 0 and new_status[id] < 0:
                    leave_boundary.append(id)   # 出界
                if self.id_status[id] < 0 and new_status[id] >= 0:
                    enter_boundary.append(id)   # 入界
        
        self.id_status = new_status
        self.result = {
            "appear_inside": appear_inside,
            "appear_outside": appear_outside,
            "disappear_inside": disappear_inside,
            "disappear_outside": disappear_outside,
            "enter_boundary": enter_boundary,
            "leave_boundary": leave_boundary
        }
        self.summary = self.gen_summary()
        return self.summary
        
    def gen_summary(self):
        # print('appear_inside:\t' + str(appear_inside))
        # print('appear_outside:\t' + str(appear_outside))
        # print('disappear_inside:\t' + str(disappear_inside))
        # print('disappear_outside:\t' + str(disappear_outside))
        # print('enter_boundary:\t' + str(enter_boundary))
        # print('leave_boundary:\t' + str(leave_boundary))
        ai, ao, di, do, enter, leave = len(self.result["appear_inside"]), len(self.result["appear_outside"]), \
            len(self.result["disappear_inside"]), len(self.result["disappear_outside"]), \
            len(self.result["enter_boundary"]), len(self.result["leave_boundary"])
        summary = f"entering: {enter} leaving: {leave} appeared: {ai + ao} disappear: {di + do}"
        return summary

