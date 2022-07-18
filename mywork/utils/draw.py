import cv2

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("ID:", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def draw_person(img, bbox_xyxy, reid_results, names, identities=None, offset=(0, 0)):
    for i, x in enumerate(bbox_xyxy):
        person_name = names[reid_results[i]]
        if str(person_name) != 'None':
            t_size = cv2.getTextSize(person_name, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            color = (0, 0, 255)
            c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
            cv2.rectangle(img, c1, c2, color, thickness=3, lineType=cv2.LINE_AA)
            cv2.rectangle(
                img, (c2[0] - t_size[0] - 3, c2[1] - t_size[1] - 4), c2, color, -1)
            cv2.putText(img, person_name, (c2[0] - t_size[0] - 3, c2[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
