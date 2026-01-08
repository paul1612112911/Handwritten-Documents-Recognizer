# https://chatgpt.com/share/67416f42-5104-800f-8de1-a521ae39145a
'''
功能概述：
圖形介面：提供一個基於 tkinter 的圖形介面，允許用戶與應用程序進行交互。
主要類別和函數：
WindowApp: 圖形介面類，負責建立視窗、管理事件和用戶交互。
MyInputDialog: 自定義的輸入對話框，允許用戶編輯識別結果。
技術細節：
畫布坐標轉換：提供 _cvt_coord 方法，在模板坐標和畫布坐標之間進行轉換，確保在不同解析度下坐標的準確性。
事件綁定：通過綁定滑鼠事件，實現了框選關注區域、編輯識別結果等功能。
多線程處理：在加載模板和進行識別時，使用線程避免界面卡頓。
'''
import numpy as np
import tkinter as tk
from cls_app import App, YoloModelPath
from tkinter import filedialog
import os
from threading import Thread
from PIL import Image, ImageTk
from glob import glob
from customtkinter import *
from time import time
from typing import Literal
from langs import TextGetter



class MyInputDialog(CTkInputDialog):
    def __init__(self, title="辨識結果修正", text="請輸入內容：", default_text="", **kwargs):
        super().__init__(title=title, text=text, **kwargs)
        self._default_text = default_text
        self.after(20, self._insert_default_text)  # 延遲插入預設文字，gpt說創建_entry屬性時被故意延遲，所以我們也要延遲。

    def _insert_default_text(self):
        self._entry.insert(0, self._default_text)  # 在此處插入預設文字

class WindowApp:
    def _cvt_coord(self, mode: Literal['tp2cv', 'cv2tp'], coord_format:str, *coords):
        """
        轉換座標點的工具函式
        以 mode 參數決定座標轉換方向，可在 'tp2cv' (模板到畫布) 與 'cv2tp' (畫布到模板) 之間切換。
        根據 coord_format 格式轉換座標，能處理 'x' 和 'y' 軸的轉換。
        確保座標轉換的準確性，特別是不同解析度和畫布大小時。
        
        :param mode: 轉換方向 'tp2cv' 或 'cv2tp'
        :param coord_format: 指定轉換的坐標軸 'x', 'y' 型別必與 coords 的配對數字相同
        :param coords: 待轉換的座標群組，可以是整數/浮點數或 np.ndarray 型態
        :return: 轉換後的座標，以列表或 np.ndarray 回傳
        """

        assert len(coord_format) == len(coords)
        assert mode == 'tp2cv' or mode == 'cv2tp'

        tp_w, tp_h = self.app.template_obj.size()
        if isinstance(coords[0], (int, float)):
            x_cvt = (lambda x: int(x*self.canvas_w/tp_w)) if mode == 'tp2cv' else (lambda x: int(x*tp_w/self.canvas_w))
            y_cvt = (lambda y: int(y*self.canvas_h/tp_h)) if mode == 'tp2cv' else (lambda y: int(y*tp_h/self.canvas_h))
        elif isinstance(coords[0], np.ndarray):
            x_cvt = (lambda x: ((x*self.canvas_w/tp_w)).astype(int)) if mode == 'tp2cv' else (lambda x: ((x*tp_w/self.canvas_w)).astype(int))
            y_cvt = (lambda y: ((y*self.canvas_h/tp_h)).astype(int)) if mode == 'tp2cv' else (lambda y: ((y*tp_h/self.canvas_h)).astype(int))
        else:
            raise TypeError('Invalid type of coords.')
        
        result = []
        for axis, coord in zip(coord_format, coords):
            if axis == 'x':
                result.append(x_cvt(coord))
            elif axis == 'y':
                result.append(y_cvt(coord))
            else:
                raise ValueError('Invalid format of coord_format.')
        
        return result
    
    def _draw_boxes_and_lines(self, canvas: CTkCanvas):
        lines = self.app.template_obj.detect_lines()
        focus_coords = [focus.coord for focus in self.app.template_obj.focuses]
        if len(lines) != 0:
            focus_rows, focus_cols = list(zip(*lines))  # list of arrays
        else:
            focus_rows, focus_cols = [],[]
        canvas.delete('detected_line', 'boxed_focus')
        for focus_coord, rows, cols in zip(focus_coords, focus_rows, focus_cols, strict=True): # 迭代每個框
            print('template rows:',rows, 'cols:', cols)
            rows, cols = self._cvt_coord('tp2cv', 'yx', rows, cols)
            print('canvas rows:',rows, 'cols:', cols)
            
            focus_coord = [self._cvt_coord('tp2cv', 'xy', *p) for p in focus_coord]
            rows += focus_coord[0][1]
            cols += focus_coord[0][0]
            
            print('focus coord on canvas:',focus_coord)
            print('rows on canvas:', rows)
            print('cols on canvas:', cols)
            for row in rows:
                canvas.create_line(cols[0], row, cols[-1], row, tags='detected_line', width=3, fill="#00f")
            for col in cols:
                canvas.create_line(col, rows[0], col, rows[-1], tags='detected_line', width=3, fill="#00f")
            
            canvas.create_rectangle(focus_coord[0][0],focus_coord[0][1],focus_coord[1][0],focus_coord[1][1], tags='boxed_focus', width=5)
    
    def _display_result_oncanvas(self, canvas: CTkCanvas, doc_num: int, tag):
        results = self.app.detect_result[doc_num]
        cell_coords = self.app.template_obj.get_indexed_coords_without_extend().copy()
        cell_coords[...,0] = self._cvt_coord('tp2cv', 'x', cell_coords[...,0])[0]
        cell_coords[...,1] = self._cvt_coord('tp2cv', 'y', cell_coords[...,1])[0]
        canvas.delete(tag)
        for result_str, coord in zip(results[0], cell_coords, strict=True):
            # print(f'display result {result_str} at {coord}')
            (x1, y1), (x2, y2) = coord
            center_x = x1 + (x2 - x1) / 2
            center_y = y1 + (y2 - y1) / 2
            canvas.create_text(center_x, center_y, text=result_str, fill='blue', font=("Arial", int((y2 - y1)*0.5)), tags=tag)

    def construct_window(self, model_names):
        
        tg = TextGetter()
        tg.set_language("Chinese") # 設定語言

        # 設定視窗大小位置
        self.root.geometry('1500x800+100+100')

        # 設定畫面長寬
        self.canvas_w = 600
        self.canvas_h = 800

        # 設定左側欄位寬度
        self.left_width = 300

        # 建立可捲動的左側欄位Frame
        left_side = CTkScrollableFrame(self.root, width=self.left_width, height=self.canvas_h)

        # 設定字體
        font = CTkFont('Microsoft JhengHei', 18)

        # 建立元件
        #  模板畫布
        template_canvas = CTkCanvas(self.root, width=self.canvas_w, height=self.canvas_h, bg='white')

        #  載入模板的按鈕
        load_template_btn = CTkButton(left_side, text=tg.get_text('LoadTemplateBtn'), font=font, state='normal')

        #  清除框的按鈕
        clear_focus_btn = CTkButton(left_side, text=tg.get_text('ClearBtn'), font=font, state='disabled')

        #  選擇使用何種模型
        model_radios = [CTkRadioButton(left_side, text=name, value=i) for i, name in enumerate(model_names)]
        choosing_model_id = tk.IntVar(value=0)
        [radio.configure(variable=choosing_model_id) for radio in model_radios]    

        #  辨識單一圖片檔案的按鈕
        detect_files_btn = CTkButton(left_side, text=tg.get_text('RecogFileDocBtn'), font=font, state='disabled')

        #  辨識資料夾內圖片的按鈕
        detect_folder_btn = CTkButton(left_side, text=tg.get_text('RecogFolderDocBtn'), font=font, state='disabled')

        # 結束編輯並儲存結果的按鈕
        end_editing_btn = CTkButton(left_side, text=tg.get_text('EndEditAndSaveBtn'), font=font, state='disabled')

        #  顯示文件圖片的畫布
        doc_canvas = CTkCanvas(self.root, width=self.canvas_w, height=self.canvas_h)
        # 建立標籤
        status_label = CTkLabel(left_side, text=tg.get_text('LoadTemplateLabel'), font=font, wraplength=self.left_width//3*2)
        # 模板畫布設定
        self.chosen_boxes = [] # 儲存以畫布為主的選擇框座標
        template_photoImage = None
        # 設定當載入模板時提供框選的畫布
        def set_template_boxing_mode():
            print('setting template canvas to boxing mode')
            if len(self.app.template_obj.focuses)==0:
                status_label.configure(text=tg.get_text('SelectRecogAreaLabel'))
            else:
                status_label.configure(text=tg.get_text('FinishSelectLabel'))

            template_canvas.delete('all')
            nonlocal template_photoImage
            template_photoImage = self.app.template_obj.to_tk_img((self.canvas_w, self.canvas_h))

            template_canvas.create_image(0,0, image=template_photoImage, anchor='nw', tags=('template'))
            choosing_points = [] # 儲存正在選擇的點座標，以canvas座標為準


            def moving_mouse(event: tk.Event):
                template_canvas.delete('cross')
                template_canvas.create_line(event.x, 0, event.x, self.canvas_h, tags=('cross'), fill='red')
                template_canvas.create_line(0, event.y, self.canvas_w, event.y, width=1, fill='#f00', tags='cross')

                # 繪製目前選擇中的眶
                if len(choosing_points) == 1:
                    template_canvas.delete('choosing_box')
                    template_canvas.create_rectangle(choosing_points[0][0], choosing_points[0][1], event.x, event.y, tags=('choosing_box'), 
                                                     outline='blue', width=3)
            
            def clear_cross(event: tk.Event): # 離開時，去除十字
                template_canvas.delete('cross')
            
            def choose_point(event: tk.Event): # 被點擊的事件
                choosing_points.append([event.x, event.y])
                if len(choosing_points) == 1:
                    status_label.configure(text=tg.get_text('SelectingLabel'))
                
                if len(choosing_points) >= 2:
                    (choosing_x1, choosing_y1), (choosing_x2, choosing_y2) = choosing_points[:2]
                    choosing_x1, choosing_x2 = sorted([choosing_x1, choosing_x2])
                    choosing_y1, choosing_y2 = sorted([choosing_y1, choosing_y2])

                    self.chosen_boxes.append(((choosing_x1, choosing_y1), (choosing_x2, choosing_y2)))

                    target_model = choosing_model_id.get()
                    choosing_x1, choosing_y1, choosing_x2, choosing_y2 = self._cvt_coord('cv2tp','xyxy', choosing_x1, choosing_y1, choosing_x2, choosing_y2)
                    self.app.template_obj.add_focus(((choosing_x1, choosing_y1), (choosing_x2, choosing_y2)), target_model)
                    choosing_points.clear()
                    
                    status_label.configure(text=tg.get_text('FinishSelectLabel'))
                    template_canvas.delete('choosing_box')
                    self._draw_boxes_and_lines(template_canvas)

                    detect_files_btn.configure(state = 'normal')
                    detect_folder_btn.configure(state = 'normal')
                    end_editing_btn.configure(state = 'disabled')
            
            template_canvas.bind("<Motion>", moving_mouse)
            template_canvas.bind("<Leave>", clear_cross)
            template_canvas.bind("<Button-1>", choose_point)

            self._draw_boxes_and_lines(template_canvas)

            clear_focus_btn.configure(state = 'normal')
            load_template_btn.configure(state = 'normal')
            if len(self.app.template_obj.focuses) == 0:
                detect_files_btn.configure(state = 'disabled')
                detect_folder_btn.configure(state = 'disabled')
            else:
                detect_files_btn.configure(state = 'normal')
                detect_folder_btn.configure(state = 'normal')
            end_editing_btn.configure(state = 'disabled')

        keep_showing_doc = None
        # 設定在模板畫布上顯示結果
        def set_template_edit_mode():
            print('setting template canvas to editable mode')
            template_canvas.unbind("<Motion>")
            template_canvas.unbind("<Leave>")
            template_canvas.unbind("<Button-1>")
            template_canvas.delete('cross')

            activating_result = 0
            cell_coords = self.app.template_obj.get_indexed_coords_without_extend().copy()
            
            status_label.configure(text=tg.get_text('RecogDoneLabel'))

            # 轉換座標
            cell_coords[...,0] = self._cvt_coord('tp2cv', 'x', cell_coords[...,0])[0]
            cell_coords[...,1] = self._cvt_coord('tp2cv', 'y', cell_coords[...,1])[0]

            # 提取座標
            left_coords = cell_coords[:,0,0]   # 左上座標的x值
            right_coords = cell_coords[:,1,0]  # 右下座標的x值
            top_coords = cell_coords[:,0,1]    # 左上座標的y值
            bottom_coords = cell_coords[:,1,1] # 右下座標的y值


            def find_target_cell_index(x, y):
                # 尋找符合游標位置的座標欄位索引值
                left_excluded_idx   = left_coords < x
                right_excluded_idx  = right_coords > x
                top_excluded_idx    = top_coords < y
                bottom_excluded_idx = bottom_coords > y
                # print(np.stack((left_excluded_idx, right_excluded_idx, top_excluded_idx, bottom_excluded_idx)))
                idx_intersect = np.where(np.all((left_excluded_idx, right_excluded_idx, top_excluded_idx, bottom_excluded_idx), axis=0))[0]

                return idx_intersect

            def hover(event: tk.Event):
                x, y = event.x, event.y
                idx_intersect = find_target_cell_index(x, y)
                if len(idx_intersect) == 1:
                    idx_intersect = idx_intersect.item()
                    
                    being_hovered_cell_coord = cell_coords[idx_intersect]
                    # print('being_hovered_cell_coord:', being_hovered_cell_coord)
                    template_canvas.delete('highlighted_cell')
                    template_canvas.create_rectangle(
                        being_hovered_cell_coord[0, 0], being_hovered_cell_coord[0, 1],
                        being_hovered_cell_coord[1, 0], being_hovered_cell_coord[1, 1],
                        outline='green', tags='highlighted_cell', width=2
                    )
                else:
                    template_canvas.delete('highlighted_cell')

            def clicked(event: tk.Event):
                x, y = event.x, event.y
                idx_intersect = find_target_cell_index(x, y)
                print('found cell index:', idx_intersect)
                if len(idx_intersect) == 1:
                    idx = idx_intersect.item()
                    chosen_result = self.app.detect_result[activating_result][0][idx] 
                    dialog = MyInputDialog(
                        title=tg.get_text('ManualInputDialogTitle'), 
                        text=tg.get_text('ManualInputDialogHint'), 
                        font=font, default_text=chosen_result
                    )
                    new_text = dialog.get_input()
                    if new_text is None:
                        new_text = chosen_result
                    self.app.detect_result[activating_result][0][idx] = new_text
                    self._display_result_oncanvas(template_canvas, activating_result, tag="recognition_result")


            self._display_result_oncanvas(template_canvas, activating_result, tag="recognition_result")
            nonlocal keep_showing_doc
            keep_showing_doc = ImageTk.PhotoImage(Image.fromarray(self.app.detect_result[activating_result][1]).resize((self.canvas_w, self.canvas_h)))
            doc_canvas.create_image(0,0,anchor=tk.NW, image=keep_showing_doc, tags='referencing_doc')

            def change_doc(event: tk.Event):
                print('event delta:',event.delta)   # event.delta: 手指往下撥為負，往上為正
                nonlocal activating_result, keep_showing_doc
                if event.delta < 0 and activating_result < len(self.app.detect_result)-1:
                    activating_result += 1
                elif event.delta > 0 and activating_result > 0:
                    activating_result -=1
                else:
                    return
                
                print(f'displaying result {activating_result}')
                self._display_result_oncanvas(template_canvas, activating_result, tag="recognition_result")
                keep_showing_doc = ImageTk.PhotoImage(Image.fromarray(self.app.detect_result[activating_result][1]).resize((self.canvas_w, self.canvas_h)))
                doc_canvas.delete('referencing_doc')
                doc_canvas.create_image(0,0,anchor=tk.NW, image=keep_showing_doc, tags='referencing_doc')                
            
            template_canvas.bind("<Motion>", hover)
            template_canvas.bind("<Button-1>", clicked) 
            template_canvas.bind("<MouseWheel>", change_doc)

            load_template_btn.configure(state = 'disabled')
            detect_files_btn.configure(state = 'disabled')
            detect_folder_btn.configure(state = 'disabled')
            clear_focus_btn.configure(state = 'disabled')
            end_editing_btn.configure(state = 'normal')
        
        def end_of_editing():
            # 消除編輯結果的數字和事件
            template_canvas.delete("recognition_result")
            template_canvas.unbind("<Motion>")
            template_canvas.unbind("<Button-1>")
            template_canvas.unbind("<MouseWheel>")
            doc_canvas.delete('referencing_doc')

            # 儲存結果
            save_path = filedialog.asksaveasfilename(
                initialdir='app2/',
                title=tg.get_text('SaveResultWindowTitle'),
                defaultextension='.xlsx',
                filetypes=[('Microsoft Excel', '*.xlsx')]
            )
            if save_path != '':
                self.app.save_results_as_excel(save_path)

            # 回到原本的框選模式
            set_template_boxing_mode()
        end_editing_btn.configure(command=end_of_editing)
        
        # 設定載入模板的按鈕
        def load_template():
            def f():
                file_path = filedialog.askopenfilename()
                if file_path == '':
                    return
                status_label.configure(text = tg.get_text('LoadingTemplateBtn'))
                self.app.load_template(file_path)
                set_template_boxing_mode()
            load_template_thread = Thread(target=f)
            load_template_thread.start()
        load_template_btn.configure(command=load_template)

        # 設定辨識檔案圖片的按鈕
        def detect_files():
            def f():
                file_pathes = filedialog.askopenfilenames()
                if len(file_pathes) == 0:
                    return
                clear_focus_btn.configure(state = 'disabled')
                load_template_btn.configure(state = 'disabled')
                detect_files_btn.configure(state = 'disabled')
                detect_folder_btn.configure(state = 'disabled')
                self.app.detect_docs(file_pathes)
                set_template_edit_mode()
            detect_files_thread = Thread(target=f)
            detect_files_thread.start()
        detect_files_btn.configure(command=detect_files)

        # 設定辨識資料夾圖片的按鈕
        def detect_folder():
            def f():
                folder_path = filedialog.askdirectory()
                if len(folder_path) == 0:
                    return
                clear_focus_btn.configure(state = 'disabled')
                load_template_btn.configure(state = 'disabled')
                detect_files_btn.configure(state = 'disabled')
                detect_folder_btn.configure(state = 'disabled')
                self.app.detect_docs(glob(folder_path+'/*'))
                set_template_edit_mode()
            detect_folder_thread = Thread(target=f)
            detect_folder_thread.start()
        detect_folder_btn.configure(command=detect_folder)


        # 設定清除框的按鈕
        def clear_focus():
            self.app.template_obj.clr_focus()
            self._draw_boxes_and_lines(template_canvas)
            detect_files_btn.configure(state = 'disabled')
            detect_folder_btn.configure(state = 'disabled')
        clear_focus_btn.configure(command=clear_focus)
        
        # 建立排版
        doc_canvas.pack(side='right')
        template_canvas.pack(side='right')
        left_side.pack(fill='y')
        load_template_btn.pack(side='top', padx=10, pady=10)
        clear_focus_btn.pack(side='top', padx=10, pady=10)
        [radio.pack(side='top', padx=10, pady=10) for radio in model_radios]
        detect_files_btn.pack(side='top', padx=10, pady=10)
        detect_folder_btn.pack(side='top', padx=10, pady=10)
        end_editing_btn.pack(side='top', padx=10, pady=10)
        status_label.pack(side='top', pady=10)


    def __init__(self, model_infos:list[list[str]]):
        self.root = CTk()
        self.app = App([YoloModelPath(*(info[1:])) for info in model_infos])
        self.construct_window([info[0] for info in model_infos])

    def start(self):
        self.root.mainloop()

