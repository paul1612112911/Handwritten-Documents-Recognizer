'''
功能概述：
模板處理與欄位管理：定義一個模板class，用於管理模板影像、關注區域（欄位），以及與欄位相關的操作。
主要類別和函數：
Focus: 表示一個關注區域，包括其坐標和對應的模型 ID。
Template: 模板類，管理模板影像、關注區域，提供方法進行欄位檢測和欄位內容提取。
技術細節：
模板初始化：在創建模板時，會自動檢測並校正模板影像，以確保其與待處理的文件一致。
關注區域管理：提供方法添加、清除關注區域，並管理每個區域使用的模型。
線條檢測：在關注區域內檢測表格的行與列，確定欄位的位置。
欄位索引化：將欄位進行索引化，便於在不同的關注區域間進行匹配和合併。
'''
import numpy as np
import cv2
from PIL import Image, ImageTk
from typing import Optional, Literal
from cv2.typing import MatLike
from detect_doc import find_mask_coordinate, find_docs
from detect_table import hough_lines_detect, adaptive_binarize
from itertools import pairwise
from nms import line_intersect, box_iou_shaped
import matplotlib.pyplot as plt
from dataclasses import dataclass

DEBUG = False

class Focus:
    def __init__(self, coord:list[list[int]], model_id:int):
        self.coord = coord
        self.model_id = model_id
    def get_dict(self):
        return {'coord':self.coord, 'model_id': self.model_id}


def internal_pad_white(img: MatLike):
    h, w = [side // 20 for side in img.shape[:2]]
    img[:h] = 255
    img[-h:] = 255
    img[:,:w] = 255
    img[:,-w:] = 255
    return img



class Template:
    def __init__(self, complete_template: MatLike, template_name: str):

        complete_template = find_docs(complete_template, 'auto')

        print('complete_template shape:', complete_template.shape)
        complete_template = cv2.cvtColor(complete_template, cv2.COLOR_BGR2GRAY)
        complete_template = adaptive_binarize(complete_template)

        complete_template = internal_pad_white(complete_template)

        self.complete_template = complete_template

        x1 = complete_template.shape[1] // 20
        x2 = complete_template.shape[1] - x1
        y1 = complete_template.shape[0] // 20
        y2 = complete_template.shape[0] - y1

        self.template = self.complete_template[y1:y2, x1:x2]

        self.focuses: list[Focus] = []
        self._gotton_focuses = False
        
        print('creating Template')
        print('complete template shape:', complete_template.shape)
        print('template shape:', self.template.shape)
        print()

        self.name = template_name

    def to_tk_img(self, resize: Optional[tuple[int, int]]=None):
        '''resize:(width, height)'''
        template = cv2.cvtColor(self.template, cv2.COLOR_GRAY2RGB)
        img_tp = Image.fromarray(template)
        if resize is not None:
            image = img_tp.resize(resize)
        else: 
            image = img_tp
        photo = ImageTk.PhotoImage(image)
        return photo
    
    def size(self):
        '''(width, height)'''
        return self.template.shape[::-1]
    
    def add_focus(self, focus_coord:list[list[int]], model_idx:int):
        self.focuses.append(Focus(focus_coord, model_idx))
        self._gotton_focuses = False
        print('now storing focuses:')
        for focus in self.focuses:
            print(focus.coord, focus.model_id)
        print()
    def set_focus(self, coords, model_indice):
        self.focuses = [Focus(coord, model_idx) for coord, model_idx in zip(coords, model_indice, strict=True)]
        self._gotton_focuses = False
    def clr_focus(self):
        self.focuses.clear()
        self._gotton_focuses = False
    def focus_coordinates(self):
        return [focus.coord for focus in self.focuses]
    
    def min_lines_distance(self):
        return min(self.template.shape[:2])*0.01
    
    def detect_lines(self) -> list[list[np.ndarray]]:
        '''
        return: [[row_border, col_border], [row_border, col_border],...]
        '''
        coordinates = self.focus_coordinates()
        result_borders = []
        for focus_coord in coordinates:
            (x1, y1), (x2, y2) = focus_coord
            focused_template = self.template[y1:y2,x1:x2]
            row_borders_per_focus, col_borders_per_focus = hough_lines_detect(focused_template, inv=True, min_gap=self.min_lines_distance())
            result_borders.append([row_borders_per_focus, col_borders_per_focus])
        return result_borders

    def borders_to_cell(self, borders_focuses:list[list[np.ndarray]], cell_extend: int|Literal['auto']) -> list[np.ndarray]:
        result_cells_focuses = []
        focuses =  [f.get_dict() for f in self.focuses]
        if cell_extend == 'auto':
            cell_extend = max(self.size()) // 400
        
        for (row_borders, col_borders), focus in zip(borders_focuses, focuses, strict=True):
            focus_left, focus_top = focus['coord'][0]
            result_per_focus = []
            if row_borders.size != 0 and col_borders.size != 0:
                for top, bottom in pairwise(row_borders):
                    for left, right in pairwise(col_borders):
                        x1 = max(left   + focus_left    - cell_extend, 0)
                        x2 = min(right  + focus_left    + cell_extend, self.template.shape[1])
                        y1 = max(top    + focus_top     - cell_extend, 0)
                        y2 = min(bottom + focus_top     + cell_extend, self.template.shape[0])
                        result_per_focus.append(((x1, y1), (x2, y2)))
            else:
                focus_right, focus_bottom = focus['coord'][1]
                result_cells_focuses.append(((focus_left, focus_top), (focus_right, focus_bottom)))
            result_cells_focuses.append(np.array(result_per_focus))

        return result_cells_focuses
    
    def index_cells(self, checking_cells_focuses: list[np.ndarray], target_cells_focuses: list[np.ndarray], focuses_shapes: list[list[int, int]]):
        '''[shaped_indice_lst, index_coordinates]'''
        coords: list[list[list[int]]] = []
        model_indice_by_coords: list[int] = []
        shaped_indice_lst: list[np.ndarray] = []
        new_idx_count = 0
        focuses_model_id: list[int] = [f.model_id for f in self.focuses]

        for i, (cells_per_focus, target_cells_per_focus, shape_per_focus, model_id) in enumerate(zip(checking_cells_focuses, target_cells_focuses, focuses_shapes, focuses_model_id)):
            
            rows, cols = shape_per_focus
            if i == 0:
                unshaped_indice = []
                for target_cell in target_cells_per_focus:
                    coords.append(target_cell)
                    unshaped_indice.append(new_idx_count)
                    new_idx_count += 1
                model_indice_by_coords.extend([model_id]*target_cells_per_focus.shape[0])

                shaped_indice_lst.append(np.array(unshaped_indice).reshape(rows, cols))
            
            else:
                unshaped_indice = np.full(rows*cols, -1, int)

                for before_cells_per_focus, before_shaped_indice, before_model_id in zip(checking_cells_focuses[:i], shaped_indice_lst, focuses_model_id[:i]):

                    (before_x1, before_y1), (before_x2, before_y2) = before_cells_per_focus[0, 0], before_cells_per_focus[-1, 1]
                    (current_x1, current_y1), (current_x2, current_y2) = cells_per_focus[0, 0], cells_per_focus[-1, 1]

                    thresh = self.min_lines_distance()
                    if max(line_intersect([before_x1, before_x2], [current_x1, current_x2]), line_intersect([before_y1, before_y2], [current_y1, current_y2])) > thresh:
                        for j, (current_cell, target_cell) in enumerate(zip(cells_per_focus, target_cells_per_focus)):
                            same_cell_idx = -1

                            for idx in before_shaped_indice.flatten():
                                if box_iou_shaped(current_cell, coords[idx]) > 0.9:
                                    same_cell_idx = idx
                                    assert before_model_id == model_id, 'there is a cell choson with different model.'
                                    break
                            
                            if same_cell_idx != -1:
                                unshaped_indice[j] = same_cell_idx

                new_cells_num = unshaped_indice[unshaped_indice==-1].size
                new_cell_coords = target_cells_per_focus[unshaped_indice == -1]
                
                unshaped_indice[unshaped_indice==-1] = np.arange(new_idx_count, new_idx_count + new_cells_num, dtype=int)
                new_idx_count += new_cells_num

                shaped_indice_lst.append(unshaped_indice.reshape(rows, cols))

                coords.extend(list(new_cell_coords))
                model_indice_by_coords.extend([model_id]*new_cells_num)

        indexed_model_id = np.array(model_indice_by_coords)


        coordinates = np.array(coords)
        return shaped_indice_lst, coordinates, indexed_model_id


    def get_cells(self):
        '''
        return: (塑型的索引值陣列組成的列表, 供給模型辨識的欄位座標, 每個欄位要給哪個模型辨識的索引)
        '''
        # 1.線條檢測
        # 2.取出欄位座標
        # 3.欄位座標索引化，檢查不同的模型的選擇框是否涵蓋同個欄位

        if self._gotton_focuses:
            return self._shaped_indice_lst, self._indexed_coordinates, self._indexed_model_id
        else:
            borders = self.detect_lines()
            
            cells_for_idx = self.borders_to_cell(borders, 0)
            cells_for_detect = self.borders_to_cell(borders, 'auto')

            print('focus shapes:')
            for focus in cells_for_idx:
                print(focus.shape)

            shapes = [[len(row)-1, len(col)-1] for row, col in borders]

            shaped_indice_lst, indexed_coordinates, indexed_model_id = self.index_cells(cells_for_idx, cells_for_detect, shapes)
            print('indexed coordinates shape:', indexed_coordinates.shape)
            _, self._indexed_cell_boxes, _ = self.index_cells(cells_for_idx, cells_for_idx, shapes)

            self._shaped_indice_lst, self._indexed_coordinates, self._indexed_model_id = shaped_indice_lst, indexed_coordinates, indexed_model_id
            self._gotton_focuses = True

        return shaped_indice_lst, indexed_coordinates, indexed_model_id
    
    def get_indexed_coords_without_extend(self):
        '''沒有擴張的欄位座標'''
        if not self._gotton_focuses:
            self.get_cells()
        return self._indexed_cell_boxes

    def center_coords(self) -> np.ndarray:
        if not self._gotton_focuses:
            self.get_cells()
        return self._indexed_coordinates.sum(axis=1)/2


