import numpy as np
import cv2
from scipy.spatial import distance as dist

def order_points(pts):
    # เรียงจุดโดยยึดแกน x
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # หยิบจุดซ้ายสุด กับ ขวาสุดมาจาก xSorted
    # เป็นพ้อย X
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # เรียงลำดับ จุดซ้ายสุด ตามแกน Y เพื่อให้ได้ top-left and bottom-left
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # ได้พิกัด ซ้ายบนมาแล้ว
    # ยึดไว้เป็นหลักในการคำนวณ ระยะทางแบบยุคลิด
    # พิกัดซ้ายบน กับ จุดขวาสุด, ตามทฤษฎีพีทาโกรัส
    # จุดที่มีระยะห่างมากที่สุด จะเป็น จุด ขวาล่าง
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # รีเทิร์นค่าพิกัด top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

def four_point_transform(image, pts):
    # รับค่าจุด
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # ทำการคำนวณ ความกว้างของรูปภาพ
    # ซึ่งจะมีระยะห่างสูงสุดระหว่าง ขวาล่าง กับ ซ้ายล่าง ตามแกน x
    # รวมถึง ขวาบน และ ซ้ายบนด้วย
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # ทำการคำนวณ ความสูงของรูปภาพ
    # ซึ่งจะมีระยะห่างสูงสุดระหว่าง ขวาบน และ ซ้ายบน ตามแกน y
    # รวมถึง ขวาล่าง กับ ซ้ายล่างด้วย
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # เราจะได้พิกัดของรูปมาแล้ว ทำการ birds eyes view
    # the top-left, top-right, bottom-right, and bottom-left order
    dst = np.array([
        [0,0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # คำนวณ perspective transform matrix แล้วนำไปใช้
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped
