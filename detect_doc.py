'''
功能概述：
提供文件檢測與透視校正的函式：在影像中檢測文件的輪廓，並進行透視變換，將文件校正為俯視圖，便於後續處理。
主要函數：
num_channels(img): 獲取影像的通道數。
gray(img_bgr): 將彩色影像轉換為灰度影像。
find_docs(img_containing_doc, complete_doc_shape): 在輸入影像中檢測文件，並進行透視變換。
技術細節：
邊緣檢測：使用 Canny 邊緣檢測找到影像中的邊緣。
輪廓檢測：利用 OpenCV 的 findContours 函數找到邊緣的輪廓。
多邊形逼近：使用 approxPolyDP 將輪廓逼近為多邊形，目標是找到有四個頂點的多邊形（即文件的四個角）。
透視變換：根據檢測到的四個頂點，計算透視變換矩陣，將文件校正為俯視圖。
'''

from typing import Literal
import numpy as np
import cv2

DEBUG = True

def num_channels(img: np.ndarray):
    if len(img.shape) == 2:
        return 1
    return img.shape[-1]

def gray(img_bgr: np.ndarray) -> np.ndarray:
    assert num_channels(img_bgr) == 3

    img_gray = np.min(img_bgr, axis=2)
    return img_gray



def find_docs(img_containing_doc: np.ndarray, complete_doc_shape: tuple[int] | Literal['auto']):
    if DEBUG:
        cv2.imwrite('debugs/find_docs img_containing_doc.png', img_containing_doc)
    
    gray_image = cv2.cvtColor(img_containing_doc, cv2.COLOR_BGR2GRAY)
    blur = gray_image
    blur = cv2.erode(blur, np.ones((71, 71)))
    blur = cv2.dilate(blur, np.ones((71, 71)))
    blur = cv2.GaussianBlur(blur, (11, 11), 1.4)
    if DEBUG:
        cv2.imwrite('debugs/find_docs blur.png', blur)

    edged = cv2.Canny(blur, 50, 90)

    if DEBUG:
        cv2.imwrite('debugs/find_docs Canny edge.png', edged)

    # 利用膨脹和侵蝕的kernel差距，填上canny edge detection可能產生的缺口
    edged = cv2.dilate(edged,np.ones((5,5)))
    edged = cv2.erode(edged, np.ones((3,3)))

    if DEBUG:
        cv2.imwrite("debugs/find_docs Canny edge after closing.png", edged)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # [[x1, y1], ...]
    # print(contours)
    contours = [contour.reshape((-1,2)) for contour in contours]
    # contour = contours[0].reshape((-1,2))
    print(contours)
    contour = max(contours,key=lambda c: (max(c, key=lambda p: p[0])[0] - min(c, key=lambda p: p[0])[0])) # 選左右最寬的輪廓
    contour = cv2.convexHull(contour)

    print('contour周長:',cv2.arcLength(contour, True)) # 周長
    torrent_rate = 0.02
    torrent_rate = 0.05
    epsilon = torrent_rate * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour,epsilon,True)
    num_angles = len(approx)

    if num_angles != 4:
        print(num_angles)
        print(contours)
        raise Exception("contour can't approx to a rectangle.")

    approx = approx.reshape((-1,2))

    if DEBUG:
        showing_img = cv2.cvtColor((gray_image.copy()), cv2.COLOR_GRAY2BGR)
        for x, y in approx:
            cv2.circle(showing_img, (x,y), 20, (255,0,0), -1)
        cv2.imwrite('debugs/find_docs four angles.png', showing_img)
        
    
    # 排序頂點順序
    rect = np.zeros((4, 2), dtype="float32")

    s = approx.sum(axis=1)
    rect[0] = approx[np.argmin(s)] # 左上
    rect[2] = approx[np.argmax(s)] # 右下

    diff = np.diff(approx, axis=1)
    rect[1] = approx[np.argmin(diff)] # 右上
    rect[3] = approx[np.argmax(diff)] # 左下

    if complete_doc_shape == 'auto':
        width = int(((rect[1,0] - rect[0,0]) + (rect[2,0] - rect[3,0])) / 2)
        height = int(((rect[2,1] - rect[1,1]) + (rect[3,1] - rect[0,1])) / 2)
    else:
        height, width = complete_doc_shape[:2]
    print(f'target doc reshape: height:{type(height)}, width:{width}')

    dst = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_containing_doc, M, (width, height))
    
    if DEBUG:
        cv2.imwrite('debugs/find_docs transformed image.png',warped)

    return warped

def find_mask_coordinate(binaried_img:np.ndarray) -> list[int]:
    """return: [x1, x2, y1, y2]"""
    coordinate = [0, 0, 0, 0]  # x1[0], x2[1], y1[2], y2[3]
    flag = 0

    if binaried_img.dtype != np.uint8:
        binaried_img = binaried_img.astype(np.uint8)

    for i, row in enumerate(255 - binaried_img):
        count = sum(row) # 檢查圖片橫排是否有非零
        if count != 0 and flag == 0: # 碰到上側
            coordinate[2] = max(i-1,0)
            flag = flag + 1
        if count != 0 and flag == 1: # 碰到下側
            coordinate[3] = min(i+1, binaried_img.shape[0]-1)

    flag = 0
    img_rotate = cv2.rotate(255 - binaried_img, cv2.ROTATE_90_CLOCKWISE) # 順時針旋轉，重複步驟，即為由左至右讀取
    for i, row in enumerate(img_rotate):
        count = sum(row)
        if count != 0 and flag == 0:
            coordinate[0] = max(i-1,0)
            flag = flag + 1
        if count != 0 and flag == 1:
            coordinate[1] = min(i+1, binaried_img.shape[1]-1)
    return coordinate

def edge_is_white(array: np.ndarray):
    # 檢查上邊
    上邊 = array[0, :]
    # 檢查下邊
    下邊 = array[-1, :]
    # 檢查左邊
    左邊 = array[:, 0]
    # 檢查右邊
    右邊 = array[:, -1]

    # 判斷四個邊是否都為 255
    return (np.all(上邊 == 255) and
            np.all(下邊 == 255) and
            np.all(左邊 == 255) and
            np.all(右邊 == 255))