import numpy as np
import cv2

def translate(image, x, y):
    # กำหนดตัวแปลง matrix และ ทำการแปลง
    M = np.float32([[1,0,x], [0,1,y]])
    shifted = cv2.warpAffine(image,M,(image.shape[1], image.shape[0]))

    return shifted

def rotate(image, angle, center=None, scale=1.0):
    # มิติของรูป
    (h,w) = image.shape[:2]

    # ถ้า จุดศุนย์กลางเป็น none จะทำการสร้างขึ้นมา
    if center is None:
        center = (w/2, h/2)

    # ทำการหมุนรูป
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w,h))

    return rotated

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # สร้างมิติของรูปภาพที่จะ resize
    dim = None
    (h,w) = image.shape[:2]

    # ถ้าทั้งความกว้างและสูงเป็น none จะคืนค่าเป็นรูปตั้งต้น
    if width is None and height is None:
        return image
    # เช็ก หากความกว้างเป็น none
    if width is None:
        # คำนวณสัดส่วนความสูง
        r = height / float(h)
        dim = (int(w*r), height)

    # เช็ก หากความสูงเป็น none
    else:
        r = width / float(w)
        dim = (width, int(h*r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized