'''
功能概述：
應用主程序：管理模板的載入、保存，文件的識別，以及結果的保存。
主要類別和函數：
App: 應用類，整合模板和文件識別的功能，提供方法供圖形介面調用。
技術細節：
模板管理：支持從圖片、PDF 或 JSON 文件中載入模板，並自動進行縮放和校正。
文件識別：遍歷待識別的文件，對每個文件進行識別，並保存識別結果。
結果保存：將識別結果保存為 Excel 文件，便於用戶查看和使用。
'''

from template import Template
from detect import DocsDetector, ExtraRuleDetector
import numpy as np
import cv2
import os
import json
from pdf2image import convert_from_path # 須對電腦進行其他軟體的安裝，才可調用此函式(讀pdf)
from typing import Optional, Callable
from openpyxl import Workbook
from glob import glob

# 1.檔案處理
# 2.整合下層其他物件
# 3.實現app功能
DEBUG = False

IMG_EXTS = ('.png', '.PNG', '.jpg', '.JPG', 'jpeg', 'JPEG')

get_stem = lambda p: os.path.splitext(os.path.basename(p))[0]

class Result:
    def __init__(self, cls_template: Template, matched_img: np.ndarray, result_strs: np.ndarray[np.str_]):
        self.cls_template = cls_template
        self.matched_img = matched_img
        self.result_strs = result_strs

class YoloModelPath:
    def __init__(self, file_path:str, dir_path:str = 'models/'):
        self.file_path = file_path
        self.dir_path = dir_path
    def path(self):
        assert os.path.exists(os.path.join(self.dir_path, self.file_path)), f"{os.path.join(self.dir_path, self.file_path)} does not exists"
        return os.path.join(self.dir_path, self.file_path)

class App:
    def __init__(self, yolo_model_paths:list[YoloModelPath]):
        self.template_obj = None # 存放Template物件
        self.template_json = None # 存放轉換自json的dict
        self.app_dir = '.' # 應用程式主要資料夾
        self.template_dir = os.path.join(self.app_dir, 'templates') # template json檔所在的資料夾
        self.complete_template_img_dir = os.path.join(self.template_dir, 'complete template images') # 處理後的template圖片檔所在資料夾
        os.makedirs(self.template_dir, exist_ok=True) # 若資料夾不存在，則創建資料夾
        self.yolo_model_paths = [ # 儲存模型的路徑
            model.path() for model in yolo_model_paths
        ]
        self.detect_result: list[tuple[np.ndarray, np.ndarray, str]] = [] # 偵測的結果  (results, matched_img, doc_name)
    
    {
        'name': str,
        'complete_template_path':str,
        'focuses':
        [
            {
                'coord':list[list[int]],
                'model_id':int
            }
        ]
    }

    def load_template(self, template_path: str): ## 先縮放圖片再創建模板物件
        if template_path.endswith(IMG_EXTS):
            complete_template = cv2.imread(template_path)

        elif template_path.endswith(('.pdf', '.PDF')):
            complete_template = np.array(convert_from_path(template_path, dpi=300)[0])
            complete_template = Template(complete_template, get_stem(template_path))

        elif template_path.endswith(('.json', '.JSON')):
            with open(template_path, 'r') as f:
                template_json = json.load(f)
            complete_template = cv2.imread(template_json['complete_template_path'], cv2.IMREAD_COLOR)
        
        else:
            raise Exception('Cannot handle the template file.')

        if template_path.endswith(('.json', '.JSON')):
            focuses_num = len(template_json['focuses'])
        else:
            focuses_num = 0


        scale_thresh = 3000
        if min(complete_template.shape[:2]) > scale_thresh and focuses_num == 0:
            
            before = complete_template.shape[:]
            if complete_template.shape[0] > complete_template.shape[1]:
                scale = scale_thresh / complete_template.shape[1]
            else:
                scale = scale_thresh / complete_template.shape[0]
            complete_template = cv2.resize(complete_template, dsize=None, fx=scale, fy=scale)
            after = complete_template.shape[:]
            print('scale template before:',before,', after:',after)
        
        self.template_obj = Template(complete_template, get_stem(template_path))

        if template_path.endswith(('json', 'JSON')):
            focuses = template_json['focuses']
            coords = [focus['coord'] for focus in focuses]
            model_indice = [focus['model_id'] for focus in focuses]
            self.template_obj.set_focus(coords, model_indice)            
        
    def save_template(self):
        complete_template_path = self.template_obj.name + '.png'
        cv2.imwrite(os.path.join(self.complete_template_img_dir, complete_template_path), self.template_obj.complete_template)
        focuses = [focus.get_dict() for focus in self.template_obj.focuses]
        name = self.template_obj.name
        json_data = {
            'name': name,
            'complete_template_path': complete_template_path,
            'focuses': focuses
        }
        json_path = os.path.join(self.template_dir, name)
        with open(json_path, 'w') as f:
            json.dump(json_data, f)

    def detect_docs(self, docs_pathes: list[str], step_callback: Optional[Callable] = None):
        extra_rule = False
        if extra_rule:
            doc_detector = ExtraRuleDetector(self.yolo_model_paths, self.template_obj, "data examples/*.txt")
        else:
            doc_detector = DocsDetector(self.yolo_model_paths, self.template_obj)

        self.detect_result = []  #  (results, matched_img, doc_name)
        for i, path in enumerate(docs_pathes):
            if step_callback is not None:
                step_callback(i, len(docs_pathes))
            img = cv2.imread(path)
            results_per_doc, matched_img = doc_detector(img)
            self.detect_result.append((results_per_doc, matched_img, get_stem(path)))
            print(f"detect result {i}:")
            print(results_per_doc)
        return self.detect_result

    def save_results_as_excel(self, save_path):
        if len(self.detect_result) == 0:
            raise Exception('there is no result.')
        
        shaped_indice_lst, _, _ = self.template_obj.get_cells()
        
        wb = Workbook()
        for indexed_results_per_doc, _, doc_name in self.detect_result:
            wb.create_sheet(doc_name)
            ws = wb[doc_name]
            print(f'saving {len(shaped_indice_lst)} focuses per doc')
            for shaped_indice in shaped_indice_lst:
                # 塑形結果
                shaped_result_per_focus = np.full_like(shaped_indice, '', "<U32")
                for i, indexed_result in enumerate(indexed_results_per_doc):
                    shaped_result_per_focus[shaped_indice==i] = indexed_result
                print('shaped result per focus:')
                print(shaped_result_per_focus)
                # 寫入excel
                for row in shaped_result_per_focus:
                    new_row = []
                    for item in row:
                        try:
                            new_row.append(float(item))
                        except ValueError:
                            new_row.append(item)
                    ws.append(new_row)
                ws.append([])
        del wb['Sheet']

        wb.save(save_path)
        if DEBUG:
            np.save(save_path+'.npy', shaped_result_per_focus.flatten())


if __name__ == '__main__':
    import debug_global_var
    models = [
        ['辨識數字的模型', 'digit.pt'],
        ['辨識英文的模型', 'lowerLetter.pt'],
        ['辨識數字和英文的混合模型', 'mix_digit-lowerLetter.pt'],
        ['辨識數字的模型(印刷體訓練)', 'digit print.pt'],
        ['辨識英文的模型(印刷體訓練)', 'lowerLetter print.pt'],
        ['辨識數字和英文的混合模型(印刷體訓練)', 'mix_digit-lowerLetter print.pt']
    ]
    table_coords = {'print':((58, 682), (1616, 1138)), 'handWritting':((64, 1361), (1638, 1814)), 'overlapping':((61, 2049), (1641, 2465))}
    
    app = App([YoloModelPath(*(info[1:])) for info in models])
    app.load_template(r"docs_detect\OCR compare/template.JPG")

    

    for have_print in [False, True][:1]:
        if have_print:
            dir = "docs_detect/OCR compare/have print training set"
        else:
            dir = "docs_detect/OCR compare/no print training set"
        for model_idx, char in enumerate(['digit', 'letter', 'mix']):
            for font in ['print', 'handWritting', 'overlapping']:
                app.template_obj.clr_focus()
                print(model_idx)
                app.template_obj.add_focus(table_coords[font], model_idx+3 if have_print else model_idx)
                debug_global_var.save_name = f"{'have print trainset' if have_print else 'no print trainset'} {font} {char}"
                if not have_print:
                    debug_global_var.pure_img_save_name = f"{font} {char}"
                else:
                    debug_global_var.pure_img_save_name = ''
                app.detect_docs([f"docs_detect/OCR compare/{char}.JPG"])
                # app.save_results_as_excel(os.path.join(dir, f"YOLO {font} {char}.xlsx"))




