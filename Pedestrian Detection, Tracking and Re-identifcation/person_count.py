# -- coding: gbk --
import numpy as np

import objtracker
from objdetector import Detector
import cv2

VIDEO_PATH = './video/test_person2.mp4'

if __name__ == '__main__':

    # ������Ƶ�ߴ磬��乩ײ�߼���ʹ�õ�polygon
    width = 1920
    height = 1080
    mask_image_temp = np.zeros((height, width), dtype=np.uint8)

    # ����һ��ײ��polygon����ɫ��
    list_pts_blue = [[204, 305], [227, 431], [605, 522], [1101, 464], [1900, 601], [1902, 495], [1125, 379], [604, 437],
                     [299, 375], [267, 289]]
    ndarray_pts_blue = np.array(list_pts_blue, np.int32)
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

    # ���ڶ���ײ��polygon����ɫ��
    mask_image_temp = np.zeros((height, width), dtype=np.uint8)
    list_pts_yellow = [[181, 305], [207, 442], [603, 544], [1107, 485], [1898, 625], [1893, 701], [1101, 568],
                       [594, 637], [118, 483], [109, 303]]
    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
    polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

    # ײ�߼���õ�mask������2��polygon����ֵ��Χ 0��1��2������ײ�߼���ʹ��
    polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2

    # ��С�ߴ磬1920x1080->960x540
    polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (width // 2, height // 2))

    # �� ɫ�� b,g,r
    blue_color_plate = [255, 0, 0]
    # �� polygonͼƬ
    blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)

    # �� ɫ��
    yellow_color_plate = [0, 255, 255]
    # �� polygonͼƬ
    yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

    # ��ɫͼƬ��ֵ��Χ 0-255��
    color_polygons_image = blue_image + yellow_image

    # ��С�ߴ磬1920x1080->960x540
    color_polygons_image = cv2.resize(color_polygons_image, (width // 2, height // 2))

    # list ����ɫpolygon�ص�
    list_overlapping_blue_polygon = []

    # list ���ɫpolygon�ص�
    list_overlapping_yellow_polygon = []

    # ��������
    down_count = 0
    # ��������
    up_count = 0

    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int((width / 2) * 0.01), int((height / 2) * 0.05))

    # ʵ����yolov5�����
    detector = Detector()

    # ����Ƶ
    capture = cv2.VideoCapture(VIDEO_PATH)

    while True:
        # ��ȡÿ֡ͼƬ
        _, im = capture.read()
        if im is None:
            break

        # ��С�ߴ磬1920x1080->960x540
        im = cv2.resize(im, (width // 2, height // 2))

        list_bboxs = []
        # ���¸�����
        output_image_frame, list_bboxs = objtracker.update(detector, im)
        # ���ͼƬ
        output_image_frame = cv2.add(output_image_frame, color_polygons_image)

        if len(list_bboxs) > 0:
            # ----------------------�ж�ײ��----------------------
            for item_bbox in list_bboxs:
                x1, y1, x2, y2, _, track_id = item_bbox
                # ײ�߼��㣬(x1��y1)��y����ƫ�Ʊ��� 0.0~1.0
                y1_offset = int(y1 + ((y2 - y1) * 0.6))
                # ײ�ߵĵ�
                y = y1_offset
                x = x1
                if polygon_mask_blue_and_yellow[y, x] == 1:
                    # ���ײ ��polygon
                    if track_id not in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.append(track_id)
                    # �ж� ��polygon list���Ƿ��д� track_id
                    # �д�track_id������Ϊ�� UP (����)����
                    if track_id in list_overlapping_yellow_polygon:
                        # ����+1
                        up_count += 1
                        print('up count:', up_count, ', up id:', list_overlapping_yellow_polygon)
                        # ɾ�� ��polygon list �еĴ�id
                        list_overlapping_yellow_polygon.remove(track_id)

                elif polygon_mask_blue_and_yellow[y, x] == 2:
                    # ���ײ ��polygon
                    if track_id not in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.append(track_id)
                    # �ж� ��polygon list ���Ƿ��д� track_id
                    # �д� track_id���� ��Ϊ�� DOWN�����У�����
                    if track_id in list_overlapping_blue_polygon:
                        # ����+1
                        down_count += 1
                        print('down count:', down_count, ', down id:', list_overlapping_blue_polygon)
                        # ɾ�� ��polygon list �еĴ�id
                        list_overlapping_blue_polygon.remove(track_id)
            # ----------------------�������id----------------------
            list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon
            for id1 in list_overlapping_all:
                is_found = False
                for _, _, _, _, _, bbox_id in list_bboxs:
                    if bbox_id == id1:
                        is_found = True
                if not is_found:
                    # ���û�ҵ���ɾ��id
                    if id1 in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.remove(id1)

                    if id1 in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.remove(id1)
            list_overlapping_all.clear()
            # ���list
            list_bboxs.clear()
        else:
            # ���ͼ����û���κε�bbox�������list
            list_overlapping_blue_polygon.clear()
            list_overlapping_yellow_polygon.clear()

        # ���������Ϣ
        text_draw = 'DOWN: ' + str(down_count) + \
                    ' , UP: ' + str(up_count)
        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                         org=draw_text_postion,
                                         fontFace=font_draw_number,
                                         fontScale=0.75, color=(0, 0, 255), thickness=2)
        cv2.imshow('Counting Demo', output_image_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
