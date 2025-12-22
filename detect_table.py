'''
功能概述：
表格線條檢測：在影像中檢測表格的橫向和縱向線條，確定表格的行與列位置。
主要函數：
merge_near_num(arr, threshold): 合併數組中相近的數值，用於合併接近的線條位置。
hough_lines_detect(focus_template, inv, min_gap): 使用霍夫變換檢測水平和垂直線條。
adaptive_binarize(gray_img): 對灰度影像進行自適應二值化處理。
技術細節：
霍夫變換：利用霍夫變換檢測影像中的直線，根據角度範圍區分水平和垂直線。
線條合併：將距離較近的線條位置進行合併，避免因多次檢測到同一線條而造成的冗餘。
自適應二值化：使用自適應閾值將灰度影像轉換為二值影像，適應光照不均的情況。
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import pairwise

DEBUG = False

def merge_near_num(arr: np.ndarray, threshold: int|float) -> np.ndarray:
    result = []
    near = []
    for n1, n2 in pairwise(arr):
        diff = abs(n1 - n2)

        if diff < threshold:
            if len(near) > 0:
                near.append(n2)
            else:
                near.extend([n1, n2])
        else:
            if len(near) > 0:
                avg = sum(near) / len(near)
                near.clear()
                result.append(avg)
            else:
                result.append(n1)
    if len(near) > 0:
        avg = sum(near) / len(near)
        result.append(avg)
    else:
        result.append(n2)   

    return np.array(result)


def hough_lines_detect(focus_template: np.ndarray, inv = False, min_gap = 0) -> tuple[np.ndarray, np.ndarray]:
    '''row_border, col_border'''
    if inv:
        focus_template = 255 - focus_template

    deg_range_r = 2  # degree range radius
    thresh = 0.75  # hough lines thresh
    deg_res = 0.5  # degree resolution

    rlines = cv2.HoughLines(focus_template, 1, np.pi * deg_res / 180, int(focus_template.shape[1]*thresh), min_theta=np.pi * ((90-deg_range_r) / 180), max_theta=np.pi * ((90+deg_range_r) / 180))

    clines1 = cv2.HoughLines(focus_template, 1, np.pi * deg_res / 180, int(focus_template.shape[0]*thresh), min_theta=np.pi * ((180-deg_range_r) / 180))
    clines2 = cv2.HoughLines(focus_template, 1, np.pi * deg_res / 180, int(focus_template.shape[0]*thresh), max_theta=np.pi * (deg_range_r/180))
    clines = np.concatenate((clines1, clines2))

    # x*cos(theta) + y*sin(theta) = rho
    row_center, col_center = [side / 2 for side in focus_template.shape]
    row_lines = np.array([(rho - row_center * np.cos(theta)) / np.sin(theta) for rho, theta in rlines.reshape(-1, 2)])
    col_lines = np.array([(rho - col_center * np.sin(theta)) / np.cos(theta) for rho, theta in clines.reshape(-1, 2)])

    row_borders: np.ndarray = row_lines.astype(np.int32)
    row_borders.sort()
    
    col_borders: np.ndarray = col_lines.astype(np.int32)
    col_borders.sort()

    print('before merged:')
    print('row:', row_borders)
    print('col:', col_borders)
    print()
    
    if isinstance(min_gap, (int, float)):
        processed_row_border = merge_near_num(row_borders, min_gap).astype(int)
        processed_col_border = merge_near_num(col_borders, min_gap).astype(int)
    elif isinstance(min_gap, (tuple, list)):
        processed_row_border = merge_near_num(row_borders, min_gap[0]).astype(int)
        processed_col_border = merge_near_num(col_borders, min_gap[1]).astype(int)
    else:
        processed_row_border = row_borders
        processed_col_border = col_borders
    
    print('after merged:')
    print('row:', processed_row_border)
    print('col:', processed_col_border)
    print()

    return processed_row_border, processed_col_border


def adaptive_binarize(gray_img: np.ndarray):
    if DEBUG:
        plt.subplot(1,2,1)
        plt.imshow(gray_img, 'gray')
        plt.title('before binarize image')
    blocksz = max(gray_img.shape) // 10
    if blocksz % 2 == 0:
        blocksz += 1
    binarized_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksz, 10)
    if DEBUG:
        plt.subplot(1,2,2)
        plt.imshow(binarized_img, 'gray')
        plt.title('binarized image')
        plt.show()
    return binarized_img
