import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import time
import warnings

import cv2
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from deep_sort import build_tracker
# from deep_sort.utils.draw import draw_boxes
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.draw import draw_person
from utils.general import check_img_size
from utils.general_2 import scale_coords, non_max_suppression
from utils.parser import get_config

warnings.filterwarnings("ignore")
img_count = 0


def bbox_r(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    # 根据绝对像素值计算相对边界框。
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


class Person_detect:
    def __init__(self, opt, source):

        # Initialize
        self.device = opt.device if torch.cuda.is_available() else 'cpu'
        self.half = self.device != 'cpu'  # half precision only supported on CUDA
        self.augment = opt.augment
        self.conf_thres = opt.conf_thres
        self.iou_thres = opt.iou_thres
        self.classes = opt.classes
        self.agnostic_nms = opt.agnostic_nms
        self.webcam = opt.cam
        # Load model
        # 加载Float32模型，确保用户设定的输入图片分辨率能整除32(如不能则调整为能整除并返回)
        # 32（下采样五次）
        self.model = attempt_load(opt.weights, map_location=self.device)
        self.model.half()  # to FP16
        # prune(self.model, 0.1)
        # Get names and colors
        # 获取类别名字
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        # 设置画框的颜色
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    def detect(self, img, im0s):
        """
            img 进行resize+pad之后的图片
            img0 原size图片
        """
        half = self.device != 'cpu'  # half precision only supported on CUDA

        # Run inference
        # 图片也设置为Float16
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # 没有batch_size的话则在最前面添加一个轴
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        """ 
            前向传播 返回pred的shape是(1, num_boxes, 5+num_class)
            h,w为传入网络图片的长和宽，注意dataset在检测时使用了矩形推理，所以这里h不一定等于w
            num_boxes = h/32 * w/32 + h/16 * w/16 + h/8 * w/8
            pred[..., 0:4]为预测框坐标
            预测框坐标为xywh(中心点+宽长)格式
            pred[..., 4]为objectiveness置信度
            pred[..., 5:-1]为分类结果
        """
        pred = self.model(img, augment=self.augment)[0]
        # Apply NMS
        """
           pred:前向传播的输出
           conf_thres:置信度阈值
           iou_thres:iou阈值
           classes:是否只保留特定的类别
           agnostic:进行nms是否也去除不同类别之间的框
           经过nms之后，预测框格式：xywh-->xyxy(左上角右下角)
           pred是一个列表list[torch.tensor]，长度为batch_size
           每一个torch.tensor的shape为(num_boxes, 6),内容为box+conf+cls
        """
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                   agnostic=self.agnostic_nms)
        bbox_xywh = []
        confs = []
        clas = []
        xy = []
        # 对每一张图片作处理
        for i, det in enumerate(pred):
            if det is not None and len(det):
                # 调整预测框的坐标：基于resize+pad的图片的坐标-->基于原size图片的坐标
                # 此时坐标格式为xy-xy
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                # 保存预测结果
                for *xyxy, conf, cls in reversed(det):
                    img_h, img_w, _ = im0s.shape  # get image shape
                    x_c, y_c, bbox_w, bbox_h = bbox_r(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append(conf.item())
                    clas.append(cls.item())
                    xy.append(xyxy)
        return np.array(bbox_xywh), confs, clas, xy


class yolo_reid:
    def __init__(self, cfg, args, path):
        self.args = args
        self.video_path = path
        use_cuda = args.use_cuda and torch.cuda.is_available()  # 使用GPU
        # Person_detect行人检测类
        self.person_detect = Person_detect(self.args, self.video_path)
        # deepsort 类
        self.deepsort = build_tracker(cfg, args.sort, use_cuda=use_cuda)
        imgsz = check_img_size(args.img_size, s=32)  # 检查输入图片大小格式，判断是否可以被32整除
        self.dataset = LoadImages(self.video_path, img_size=imgsz)  # imgsz=1088
        self.query_feat = np.load(args.query)
        self.names = np.load(args.names)  # 加载person和name特征向量

    def deep_sort(self):
        idx_frame = 0
        for video_path, img, ori_img, vid_cap in self.dataset:  # 读取所有的视频帧
            start = time.time()
            idx_frame += 1
            if idx_frame % 2 == 0:
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)
                bbox_xywh, cls_conf, cls_ids, xy = self.person_detect.detect(img, ori_img)
                if bbox_xywh.all() and cls_conf:
                    # do tracking  # features:reid模型输出512dim特征
                    outputs, features = self.deepsort.update(bbox_xywh, cls_conf, ori_img)
                    print(f'\033[1;35m{len(outputs)}  {len(bbox_xywh)}\033[0m \033[1;34m{features.shape}\033[0m')
                    person_cos = cosine_similarity(features, self.query_feat)
                    max_idx = np.argmax(person_cos, axis=1)
                    maximum = np.max(person_cos, axis=1)
                    max_idx[maximum < 0.52] = -1
                    reid_results = max_idx
                    draw_person(ori_img, xy, reid_results, self.names)  # 显示行人的姓名
            if self.args.display:
                end = time.time()
                seconds = end - start
                if seconds != 0:
                    fps = 1 / seconds
                    fps = "%.2f fps" % fps
                    cv2.putText(ori_img, fps, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.imshow("Person-Reid", ori_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default='./video/reid_3.mp4', type=str)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # yolov5
    parser.add_argument('--weights', nargs='+', type=str, default='weights/lite-g-strong.pt',
                        help='onnx filepath')
    # 输入图片大小
    parser.add_argument('--img-size', type=int, default=1088, help='inference: size (pixels)')
    # 置信度阈值
    parser.add_argument('--conf-thres', type=float, default=0.34, help='object confidence threshold')
    # 做nms(非极大抑制）的iou值
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    # 设置检测的类
    parser.add_argument('--classes', default=[0], type=int, help='filter by class: --class 0, or --class 0 2 3')
    # 进行nms是否也去除不同类别之间的框，默认False
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # 推理的时候进行多尺度，翻转等操作(TTA（测试时的数据增强)）推理
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    # deep_sort
    parser.add_argument("--sort", default=False, help='True: sort model or False: reid model')
    parser.add_argument("--config_deepsort", type=str, default="./deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--display", default=True, help='show result')
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)

    # reid
    parser.add_argument("--query", type=str, default="./fast_reid/query/query_features.npy")
    parser.add_argument("--names", type=str, default="./fast_reid/query/names.npy")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)  # config_file是指定的yaml配置文件，通过merge_from_file这个函数会将yaml文件中指定的超参数对默认值进行覆盖。
    yolo_reid = yolo_reid(cfg, args, path=args.video_path)
    with torch.no_grad():  # 推理时不跟踪梯度信息减少缓存浪费
        yolo_reid.deep_sort()
