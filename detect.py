'''
功能概述：
文件識別主流程：使用模板對輸入的文件進行匹配，提取欄位內容，並使用 YOLO 模型進行內容識別。
主要類別和函數：
DocsDetect: 文件檢測類，負責整合模板匹配、欄位提取、內容識別的流程。
draw_bounding_boxes(detections, image, alphabet): 在影像上繪製預測的邊界框和分類標籤。
技術細節：
模板匹配：使用 cv2.matchTemplate 將輸入的文件影像與模板進行匹配，找到模板在文件中的位置。
表格線條去除：在提取的欄位影像中，去除可能存在的表格線條，避免影響內容識別。
欄位內容識別：對每個欄位影像，使用對應的 YOLO 模型進行預測，識別其中的字符或數字。
結果處理：將模型的預測結果進行排序和合併，生成最終的識別字符串。
'''
import numpy as np
import cv2
from cv2.typing import MatLike
from template import Template
from ultralytics import YOLO
from detect_doc import edge_is_white, find_docs
from detect_table import adaptive_binarize
from nms import nms
from revision import RevisionModel

import debug_global_var

DEBUG = True

DIGITS = '0123456789.'
LOWER_LETTERS = 'abcdefghijklmnopqrstuvwxyz'

def draw_bounding_boxes(detections, image, alphabet):
    """
    在圖片上畫出邊界框並標記分類名稱（cls），不顯示 conf。
    
    :param detections: list，格式為 [[cls, box_xyxy, conf], [cls, box_xyxy, conf], ...]，
                       其中 cls 是分類名稱（字串），box_xyxy 是 [x_min, y_min, x_max, y_max]。
    :param image: numpy.ndarray，輸入圖片。
    :return: numpy.ndarray，標記了邊界框的圖片。
    """
    # 複製圖片以避免修改原圖
    image_with_boxes = image.copy()

    for detection in detections:
        cls, box_xyxy, conf = detection
        x_min, y_min, x_max, y_max = map(int, box_xyxy)
        
        # 設定顏色與框線粗細
        color = (0, 255, 0)  # 綠色框
        thickness = 2
        
        # 繪製邊界框
        cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), color, thickness)
        
        # 在框上方標記分類名稱
        label = alphabet[cls]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_x = x_min
        text_y = y_min - 5 if y_min - 5 > 10 else y_min + 15  # 確保文字不超出圖片邊界
        
        # 畫文字背景
        cv2.rectangle(image_with_boxes, 
                      (text_x, text_y - text_size[1]), 
                      (text_x + text_size[0], text_y), 
                      color, 
                      -1)
        
        # 畫文字
        cv2.putText(image_with_boxes, 
                    label, 
                    (text_x, text_y - 2), 
                    font, 
                    font_scale, 
                    (0, 0, 0),  # 黑色文字
                    font_thickness, 
                    lineType=cv2.LINE_AA)
    
    return image_with_boxes

class DocsDetector:
    def __init__(self, model_paths: list[str], cls_template: Template):
        self.models = [YOLO(p) for p in model_paths]
        self.model_paths = model_paths
        self.cls_template = cls_template
        self.shaped_indice_lst, self.coordinates, self.indexed_model_id = self.cls_template.get_cells()
    
    def iter_model(self):
        for model_path in self.model_paths:
            yield YOLO(model_path)
        
    def _template_match(self, doc_containing_img: MatLike, docs_location: bool = True):

        if docs_location and not edge_is_white(cv2.threshold(cv2.cvtColor(doc_containing_img, cv2.COLOR_BGR2GRAY),200,255,cv2.THRESH_BINARY)[1]):
            print('perspective distortion')
            doc_img = find_docs(doc_containing_img, self.cls_template.complete_template.shape)
        else:
            doc_img = cv2.resize(doc_containing_img.copy(), self.cls_template.complete_template.shape[1::-1])
        
        # 灰階化
        doc_img = cv2.cvtColor(doc_img, cv2.COLOR_BGR2GRAY)

        # 二值化
        doc_img = adaptive_binarize(doc_img)
    
        assert doc_img.shape == self.cls_template.complete_template.shape, f'doc shape{doc_img.shape} != complete template shape{self.cls_template.complete_template.shape}'

        TM_method = cv2.TM_SQDIFF # template match method

        template = self.cls_template.template
        if docs_location:
            horizontal_scalings = [-5, 0, 5]
            vertical_scalings = [-5, 0, 5]
            locs = []
            for vscal in vertical_scalings:
                for hscal in horizontal_scalings:
                    scaled_shape = (doc_img.shape[1] + hscal, doc_img.shape[0] +vscal)
                    scaled_img = cv2.resize(doc_img, scaled_shape) # x, y
                    result = cv2.matchTemplate(scaled_img, template, TM_method) 
                    if TM_method != cv2.TM_SQDIFF:
                        locs.append([scaled_shape, np.unravel_index(np.argmax(result, axis=None), result.shape)[::-1], np.max(result)]) # [img.shape, (x, y), matching_value], [::-1] 作用：(y,x) -> (x,y)
                    else:
                        locs.append([scaled_shape, np.unravel_index(np.argmin(result, axis=None), result.shape)[::-1], np.max(result)]) # [img.shape, (x, y), matching_value], [::-1] 作用：(y,x) -> (x,y)

            shape, loc, _ = max(locs, key = lambda x: x[2])
        
        else:
            result = cv2.matchTemplate(doc_img, template, TM_method)
            if TM_method != cv2.TM_SQDIFF:
                loc = np.unravel_index(np.argmax(result, axis=None), result.shape)[::-1]  # [::-1] 作用：(y,x) -> (x,y)
            else:
                loc = np.unravel_index(np.argmin(result, axis=None), result.shape)[::-1]
        
        template_h, template_w = template.shape[:2]
        matched_image = doc_img[loc[1]:loc[1]+template_h, loc[0]: loc[0]+template_w]
        matched_image = cv2.threshold(matched_image, 128, 255, cv2.THRESH_BINARY)[1] # 透視變換後，重新二值化
        
        print()
        print('matched location:', ((loc[0], loc[0]+template_w), (loc[1], loc[1]+template_h)))
        print('complete template shape:', self.cls_template.complete_template.shape)
        print('doc image shape:', doc_img.shape)
        print('template shape:', template.shape)
        print('matched doc shape:', matched_image.shape)
        print()

        if DEBUG:
            cv2.imwrite('debugs/TM_croppedTemplate.png', template)
            cv2.imwrite('debugs/TM_docImg.png', doc_img)
            cv2.imwrite('debugs/TM_matchedImg.png', matched_image)

            canvas = doc_img.copy()
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(canvas, (loc[0], loc[1]), (loc[0]+template_w, loc[1]+template_h), (0, 255, 0), 4)
            cv2.imwrite('debugs/TM_drawnTmplate.png', canvas)
            
        return matched_image

    # 偵測四方表格線，若有表格線則連同表格線和此線範圍以外塗白
    @staticmethod
    def _remove_table_line(binary_img: np.ndarray, threshold: float = 1.0, erase_thickness: int = 2) -> np.ndarray:

        # 獲取陣列的行數和列數
        rows, cols = binary_img.shape

        # 計算中心點座標
        row_center = rows // 2
        col_center = cols // 2

        # 計算行和列的閾值長度
        row_threshold = int(rows * threshold)
        col_threshold = int(cols * threshold)
        
        # 定義檢查一行或一列是否有超過閾值長度的連續0的函式
        def mark_zeros_line(line, threshold):
            max_count = 0
            current_count = 0
            for val in line:
                if val == 0:
                    current_count += 1
                else:
                    max_count = max(max_count, current_count)
                    current_count = 0
            max_count = max(max_count, current_count)
            return max_count >= threshold
        
        # 複製原始陣列
        result = binary_img.copy()
        
        # 從中間分別往四方檢查
        # 上
        for i in range(row_center, -1, -1):
            if mark_zeros_line(binary_img[i, :], col_threshold):
                result[:erase_thickness + i + 1] = 255
                break
        # 下
        for i in range(row_center, rows):
            if mark_zeros_line(binary_img[i, :], col_threshold):
                result[i-erase_thickness:] = 255
                break
        # 左
        for i in range(col_center, -1, -1):
            if mark_zeros_line(binary_img[:, i], row_threshold):
                result[:, :i+erase_thickness+1] = 255
                break
        # 右
        for i in range(col_center, cols):
            if mark_zeros_line(binary_img[:, i], row_threshold):
                result[:, i-erase_thickness:] = 255
                break

        # 返回處理後的陣列
        return result

    def _get_cells_img(self, matched_img: np.ndarray, cell_coordinate: np.ndarray, remove_border: int | str = 'auto') -> np.ndarray:
        (x1,y1),(x2,y2) = cell_coordinate
        print('processing cell img coord:', cell_coordinate, cell_coordinate.dtype)
        img = matched_img[y1:y2, x1:x2]
        # 去除表格線
        if remove_border != 0:
            if remove_border == 'auto':
                erase = min((x2-x1), (y2-y1)) // 10
                img = self._remove_table_line(img, 0.7, erase)
            else:
                img = self._remove_table_line(img,0.7,remove_border)
        
        # 補成正方形
        if img.shape[0] > img.shape[1]:
            padding = (img.shape[0] - img.shape[1]) // 2
            padded_img = np.full((img.shape[0],img.shape[0]),255,np.uint8)
            padded_img[:,padding: padding + img.shape[1]] = img

        elif img.shape[0] < img.shape[1]:
            padding = (img.shape[1] - img.shape[0]) // 2
            padded_img = np.full((img.shape[1],img.shape[1]),255,np.uint8)
            padded_img[padding:padding+img.shape[0]] = img
        else:
            padded_img = img

        img = cv2.cvtColor(padded_img,cv2.COLOR_GRAY2BGR) 
        return img


    def postprocess(self, boxes):
        return nms(boxes, 0.8)
    
    def model_init(self, model) -> None:
        pass

    def __call__(self, unmatched_img: MatLike):

        matched_img = self._template_match(unmatched_img, docs_location=True)
        cell_results = np.full(self.coordinates.shape[0], '', dtype='<U32')

        for i, model in enumerate(self.models):
            print('indexed model id shape:',self.indexed_model_id.shape)
            print('match model boolean shape:',(self.indexed_model_id == i).shape)
            print('coordinates shape:',self.coordinates.shape)

            target_coordinates_per_model = self.coordinates[self.indexed_model_id == i].reshape(-1,2,2)
            print('target model coordinates shape:',target_coordinates_per_model.shape)
            processed_cells_imgs = [self._get_cells_img(matched_img, coord) for coord in target_coordinates_per_model]

            # for ocr detect
            # if DEBUG:
            #     import os
            #     for j, img in enumerate(processed_cells_imgs):
            #         saving_img = img.copy()
            #         saving_img = cv2.resize(saving_img, (192, 192))
            #         saving_path = rf'../../overlapnet paper continue\datasets/my datasets\real nodot handwritting\single digits\fetched fields/{j}.png'
            #         cv2.imwrite(saving_path, saving_img)

            if len(processed_cells_imgs) != 0:
                model_results = model(processed_cells_imgs)
            else:
                model_results = []

            name_class: dict = model.names
            self.model_init(model)
            model_result_strings = []

            for j, result in enumerate(model_results):
                classes = result.boxes.cls.cpu().numpy().astype(np.int32)
                bbox_xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                sorted_boxes = sorted(zip(classes, bbox_xyxy, confs), key=lambda x: x[1][0] + x[1][2]) # 以x座標中心點排序

                sorted_boxes = self.postprocess(sorted_boxes)  # 未來：如果同時遇到相似字母，則啟動邏輯判斷

                # name_class[box[0]] if '10' != name_class[box[0]] != 'point' else '.' 是在訓練時標示分類命名的yaml檔在小數點的命名不一致的關係，在此做小數點的轉換。如果類別名是'10'或'point'則在偵測結果顯示'.'。

                model_result_strings.append(''.join([name_class[box[0]] if '10' != name_class[box[0]] != 'point' else '.' for box in sorted_boxes]))

            print('result of', debug_global_var.save_name)
            print(F"model {i} result:{model_result_strings}")
            cell_results[self.indexed_model_id == i] = model_result_strings
        return cell_results, matched_img

from glob import glob
class ExtraRuleDetector(DocsDetector):
    def __init__(self, model_paths, cls_template, shell_style_path: str):
        super().__init__(model_paths, cls_template)
        self.revision = RevisionModel(2)
        self.train_revision_model(shell_style_path)

    def model_init(self, model:YOLO): # override
        self.revision.setNames(model.names)

    def postprocess(self, boxes): # override
        return self.revision.revise_yoloResult(boxes, 0) # 未來擴充成可選規則

    def train_revision_model(self, shell_style_path: str):
        train_data: list[list[str]] = []
        for path in glob(shell_style_path):
            with open(path, 'r') as f:
                train_data.append([s.strip() for s in f.readlines()])
        self.revision.fit(train_data)

