import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.append('..')
from fast_reid.fastreid.config import get_cfg
from predictor import FeatureExtractionDemo


# 超参数配置
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()  # 获取已经配置好默认参数的cfg
    # config_file是指定的yaml配置文件，
    # 通过merge_from_file这个函数会将yaml文件中指定的超参数对默认值进行覆盖
    cfg.merge_from_file(args.config_file)
    # merge_from_list作用同上面的类似，只不过是通过命令行的方式覆盖
    cfg.merge_from_list(args.opts)
    # freeze函数的作用是将超参数值冻结，避免被程序不小心修改
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        default=r'D:\YOLO\yolov5-deepsort\kd-r34-r101_ibn\config-test.yaml',
        help="path to config file",
    )
    parser.add_argument(
        "--parallel",
        action='store_true',
        help='If use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--input",
        # nargs="+",
        default='../tools/deploy/test_data/0065_c6s1_009501_02.jpg',
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='demo_output',
        help='path to save features'
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', r'D:\YOLO\yolov5-deepsort\kd-r34-r101_ibn\model_final.pth'],
        nargs=argparse.REMAINDER,
    )
    return parser


class Reid_feature:
    def __init__(self):
        args = get_parser().parse_args()
        cfg = setup_cfg(args)  # 获取配置的参数
        self.demo = FeatureExtractionDemo(cfg, parallel=args.parallel)  # 构造一个demo实例

    def __call__(self, img_list):
        import time
        time.time()
        features = self.demo.run_on_image(img_list)
        return features  # 返回提取的特征


def cosin_metric(x1, x2):  # (4, 512)*(512,)
    return np.dot(x1, x2) / (np.linalg.norm(x1, axis=1) * np.linalg.norm(x2))


if __name__ == '__main__':
    start = time.time()
    Reid_feature = Reid_feature()  # 创建类的实例
    embeddings = []
    names = []
    embeddings = np.ones((1, 512), dtype=np.int)
    for person_name in os.listdir('../query'):
        if not os.path.isdir(os.path.join('../query', person_name)):
            continue
        for image_name in os.listdir(os.path.join('../query', person_name)):
            # 读取query中的person姓名和图片
            img = cv2.imread(os.path.join('../query', person_name, image_name))
            t1 = time.time()
            feat = Reid_feature(img)  # normalized feat 标准化特征图
            # print('pytorch time:', time.time() - t1)
            pytorch_output = feat.numpy()
            embeddings = np.concatenate((pytorch_output, embeddings), axis=0)
            names.append(person_name)
    names = names[::-1]
    names.append("None")
    # print(embeddings[:-1, :].shape, names)

    # 保存行人特征和names，方便进行特征比对，识别出对应的names。
    np.save(os.path.join('../query', 'query_features'), embeddings[:-1, :])
    np.save(os.path.join('../query', 'names'), names)  # save query
    query = np.load('../query/query_features.npy')
    end = time.time()
    print('Running time: %s Seconds' % (end - start))
    # cos_sim = cosine_similarity(embeddings, query)
    # print(cos_sim)
    # max_idx = np.argmax(cos_sim, axis=1)
    # maximum = np.max(cos_sim, axis=1)
    # max_idx[maximum < 0.6] = -1
    # score = maximum
    # results = max_idx
    # print(score, results)
    # for i in range(4):
    #     label = names[results[i]]
    #     print(label)
