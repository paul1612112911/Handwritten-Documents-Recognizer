'''
功能概述：
Non-Maximum Suppression (NMS) 的實現，用於在對象檢測中消除重疊的預測框，保留最有可能的預測。
主要函數：
line_intersect(line1, line2): 計算兩條線段（一維區間）的重疊長度。
is_shape_2by2(lst): 檢查輸入是否為 2x2 的列表（矩形的兩個點）。
box_iou_shaped(box1, box2): 計算兩個矩形（以 [xy, xy] 形式給出）的 IoU（交併比）。
box_iou_flatten(box1, box2): 計算兩個矩形（以 [x1, y1, x2, y2] 形式給出）的 IoU。
nms(boxes, iou_thres): 實現 NMS 演算法，根據 IoU 閾值過濾重疊的預測框。
技術細節：
IoU 計算：通過計算交集和聯集的面積，得到兩個矩形之間的 IoU 值。
NMS 演算法：比較每一對預測框，如果它們的 IoU 超過閾值，則保留置信度較高的框。
'''

import numpy as np
def line_intersect(line1: list, line2: list) -> int:
    l_max = min(line1[1], line2[1])
    l_min = max(line1[0], line2[0])
    if l_max <= l_min:
        return 0
    intersection = l_max - l_min
    return intersection
def is_shape_2by2(lst):
    import os
    # 確保是個2x2的列表
    if isinstance(lst, (list, tuple)) and len(lst) == 2:
        for sub_lst in lst:
            if not (isinstance(sub_lst, (list, tuple)) and len(sub_lst) == 2):
                return False
            for item in sub_lst:
                if not isinstance(item, int):
                    return False
        return True
    return False
def box_iou_shaped(box1, box2):
    '''
    box:[xy,xy]
    '''
    if isinstance(box1, np.ndarray):
        assert box1.shape == (2,2)
        box1 = box1.tolist()
    else:
        assert is_shape_2by2(box1)
    if isinstance(box2, np.ndarray):
        assert box2.shape == (2,2)
        box2 = box2.tolist()
    else:
        assert is_shape_2by2(box2)
    
    box1 = box1[0]+box1[1]
    box2 = box2[0]+box2[1]
    return box_iou_flatten(box1, box2)

def box_iou_flatten(box1: np.ndarray, box2: np.ndarray) -> float:
    '''
    box:xyxy
    '''
    x_max = min(box1[2],box2[2])
    x_min = max(box1[0], box2[0])
    if x_max < x_min: 
        return 0.0
    y_max = min(box1[3],box2[3])
    y_min = max(box1[1], box2[1])
    if y_max < y_min:
        return 0.0
    intersection = (x_max - x_min) * (y_max - y_min)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union

def nms(boxes: list, iou_thres: float) -> list:
    """
    boxes:[[cls, box_xyxy, conf], [cls, box_xyxy, conf], ...]
    """
    boxes = boxes[:]
    num_boxes = len(boxes)
    remove_idx = []
    for i in range(num_boxes):
        for j in range(i):
            # i < j
            if box_iou_flatten(boxes[i][1],boxes[j][1]) > iou_thres:
                if boxes[i][2] > boxes[j][2]:
                    remove_idx.append(i)
                else:
                    remove_idx.append(j)
    remove_idx.sort(reverse=True)
    for box in remove_idx:
        boxes.pop(box)
    return boxes



