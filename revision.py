from typing import Sequence
from nms import box_iou_flatten
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def classify(original_char:str):
    """
    0: padding
    1: 數字
    2: 大寫字母
    3: 小寫字母
    4: 小數點
    """
    assert len(original_char) == 1, 'only 1 char str.'
    if original_char == ' ':
        return 0
    if original_char.isdigit():
        return 1
    if original_char.isalpha():
        if original_char.isupper():
            return 2
        if original_char.islower():
            return 3
    if original_char == '.':
        return 4
    
    raise ValueError(f"no matched class: {original_char}")

name_class = ['填充', '數字', '大寫英文', '小寫英文', '小數點']


def classify_str(original_str:str, maxlen=0) -> list[int]:
    if maxlen != 0:
        if len(original_str) <= maxlen:
            # print(original_str)
            # print("{:" + str(maxlen) + "s}")
            original_str = ("{:" + str(maxlen) + "s}").format(original_str)
        else:
            original_str = original_str[len(original_str) - maxlen :]
    return [classify(c) for c in original_str]


def onehot_encode(classes:list[int], num_cls:int = 5) -> list[int]:
    result_encode = []
    for cls in classes:
        encode = [0] * num_cls
        encode[cls] = 1
        result_encode.extend(encode)
    return result_encode

def process_yolo_output(yolo_boxes:list, iou_thres: float) -> list[list[int | list[int, float]]]:
    """
    yolo_boxes: [[cls, box_xyxy, conf], [cls, box_xyxy, conf], ...]

    return: list[[cls, box_xyxy, conf](確定的) | list[[cls, box_xyxy, conf](不確定的)]
    """
    num_boxes = len(yolo_boxes)
    position_set = defaultdict(list)
    for i in range(num_boxes):
        same_position = False
        for j in range(i):
            if box_iou_flatten(yolo_boxes[i][1],yolo_boxes[j][1]) > iou_thres:
                position_set[j].append(i)
                same_position = True
                break
        if not same_position:
            position_set[i].append(i)
    positions = [p[1] for p in sorted(position_set.items(), key=lambda x: x[0])]
    return [yolo_boxes[p[0]] if len(p) == 1 else [yolo_boxes[k] for k in p] for p in positions]


class RevisionModel:
    def __init__(self, window_size: int):
        self.model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
        self.window_size = window_size
        self.names = None
        self.classification_transform = None
        self.type_num = None
    
    def _has_trained(self):
        return self.type_num is not None
        
    def setNames(self, names: dict):
        self.names = {k: names[k] if '10' != names[k] != 'point' else '.' for k in names.keys()} # 轉換小數點
        
        # 字元分類(cls) -> 決策樹分類
        self.classification_transform = {k: classify(names[k]) if '10' != names[k] != 'point' else 4 for k in names.keys()}
        pass

    def fit(self, train_data: Sequence[Sequence[str]]):
        train_x = []
        train_y = []
        type_num = len(train_data)
        self.type_num = type_num
        for data_type, strings in enumerate(train_data):
            type_vec = onehot_encode([data_type], type_num)
            for s in strings:
                classes = classify_str(s)
                for i, c in enumerate(classes):
                    if i < self.window_size:
                        train_x.append(type_vec + [i] + onehot_encode([0]*(self.window_size - i) + classes[:i]))
                    else:
                        train_x.append(type_vec + [i] + onehot_encode(classes[i-self.window_size:i]))
                    train_y.append(c)
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        print(train_x)
        print(train_y)
        self.model.fit(train_x, train_y)
    def predict(self, determined_str:str, data_type:int):
        assert self._has_trained()
        x = np.array(onehot_encode([data_type], self.type_num) + [len(determined_str)] + onehot_encode(classify_str(determined_str, maxlen=self.window_size))).reshape(1, -1)
        return self.model.predict(x)
    
    def revise_yoloResult(self, boxes, data_rule_id:int):
        """
        boxes: [[cls, box_xyxy, conf], [cls, box_xyxy, conf], ...]
        return: [[cls, box_xyxy, conf], [cls, box_xyxy, conf], ...]
        """
        if self.names is None or self.classification_transform is None:
            raise RuntimeError("no names")
        
        boxes = process_yolo_output(boxes, 0.8)

        result_boxes = []
        for i, box in enumerate(boxes):
            if not isinstance(box[0], Sequence):
                result_boxes.append(box)
                continue
            determined_str = ''.join([self.names[b[0]] for b in result_boxes])
            predicted_classification = self.predict(determined_str, data_rule_id).item()
            predicted_boxes = []
            for candidate_box in box:
                # print(possible_box)
                if self.classification_transform[candidate_box[0]] == predicted_classification:
                    predicted_boxes.append(candidate_box)
            if len(predicted_boxes) == 0:
                result_boxes.append(max(box, key=lambda b: b[2]))
            else:
                result_boxes.append(max(predicted_boxes, key=lambda b: b[2]))
        return result_boxes


if __name__ == "__main__":
    boxes = [
        [0,[1,1,10,10],0.8],
        [1,[1,1,10,10],0.9],
        [2,[20,1,30,10],0.7],
    ]
    train_data = [
        [
            "ab123",
            "cd4567",
            "efg8901"
        ],
        [
            "0.123",
            "2.345",
            "34.567",
            "89.012"
        ]
    ]
    model = RevisionModel(2)
    model.fit(train_data)
    indice = [i for i in range(10 + 1 + 26)]
    names = [str(i) for i in range(11)] + [chr(c) for c in range(ord('a'), ord('z')+1)]
    names_dict = {k:v for k, v in zip(indice, names, strict=True)}
    model.setNames(names_dict)
    print(model.predict("1", 1))

