import os

import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.torch_utils import select_device

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import objtracker

OBJ_LIST = ['person']  # 检测器要检测的目标
DETECTOR_PATH = 'weights/v5lite-g.pt'  # 检测器m比s精度高，但是速度稍慢


class baseDet(object):
    def __init__(self):
        self.img_size = 640
        self.threshold = 0.3
        self.stride = 1

    def build_config(self):
        self.frameCounter = 0

    def feedCap(self, im, func_status):  # im是视频帧
        retDict = {  # 返回检测后的字典
            'frame': None,
            'list_of_ids': None,
            'obj_bboxes': []
        }
        self.frameCounter += 1  # 帧计数
        im, obj_bboxes = objtracker.update(self, im)  # 调用deepsort类中的update
        retDict['frame'] = im
        retDict['obj_bboxes'] = obj_bboxes
        # 得到可以直接在原图上绘制的目标框
        return retDict

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def detect(self):
        raise EOFError("Undefined model type.")


class Detector(baseDet):  # 对YOLOv5检测器的封装
    def __init__(self):
        super(Detector, self).__init__()
        self.init_model()
        self.build_config()

    def init_model(self):
        self.weights = DETECTOR_PATH
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)  # 加载权重
        model.to(self.device).eval()
        model.half()
        self.m = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names

    def preprocess(self, img):  # 预处理图片
        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]  # 调整图片尺寸
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()  # 半精度
        img /= 255.0  # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img0, img  # img0原始图像，img经变换后的图像

    def detect(self, im):
        im0, img = self.preprocess(im)  # 预处理
        pred = self.m(img, augment=False)[0]  # img传入模型进行推理
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.4)  # 结果非极大抑制
        pred_boxes = []
        for det in pred:  # 对pred的检测做迭代
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    if not lbl in OBJ_LIST:
                        continue  # 是否在OBJ_LIST中
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))
        return im, pred_boxes
