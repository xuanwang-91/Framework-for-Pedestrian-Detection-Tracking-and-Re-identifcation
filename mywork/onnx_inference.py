import argparse
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime


def get_parser():
    parser = argparse.ArgumentParser(description="onnx model inference")

    parser.add_argument(
        "--model-path",
        default="outputs/onnx_model/baseline_R50.onnx",
        help="onnx model path"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.png'",
    )
    parser.add_argument(
        "--output",
        default='onnx_output',
        help='path to save converted caffe model'
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="height of image"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="width of image"
    )
    return parser


def preprocess(image_path, image_height, image_width):
    original_image = cv2.imread(image_path)
    # the model expects RGB inputs
    original_image = original_image[:, :, ::-1]

    # Apply pre-processing to image.
    img = cv2.resize(original_image, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
    img = img.astype("float32").transpose(2, 0, 1)[np.newaxis]  # (1, 3, h, w)
    return img


def normalize(nparray, order=2, axis=-1):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)


if __name__ == "__main__":
    start = time.time()
    args = get_parser().parse_args()
    embeddings = []
    names = []
    embeddings = np.ones((1, 512), dtype=np.int_)
    ort_sess = onnxruntime.InferenceSession(args.model_path)
    input_name = ort_sess.get_inputs()[0].name
    # print(os.listdir(r'fast_reid/query'))
    for person_name in os.listdir(r'fast_reid/query'):
        if not os.path.isdir(os.path.join(r'fast_reid/query', person_name)):
            continue
        for image_name in os.listdir(os.path.join(r'fast_reid/query', person_name)):
            path = os.path.join(r'fast_reid/query', person_name, image_name)
            image = preprocess(path, args.height, args.width)
            feat = ort_sess.run(None, {input_name: image})[0]
            feat = normalize(feat, axis=1)
            embeddings = np.concatenate((feat, embeddings), axis=0)
            names.append(person_name)
    names = names[::-1]
    names.append("None")
    # print(embeddings[:-1, :].shape, names)
    np.save(os.path.join(r'fast_reid/query', 'query_features'), embeddings[:-1, :])
    np.save(os.path.join(r'fast_reid/query', 'names'), names)  # save query
    end = time.time()
    print('Running time: %s Seconds' % (end - start))
