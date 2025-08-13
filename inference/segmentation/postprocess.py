# 图像预测模块
import cv2
import numpy as np


# 对其操作
# def aligning(mask,point):
#     # 获取最小外接矩形
#     rect=cv2.minAreaRect(point)
#     # 获取矩形四个点
#     box=np.intp(cv2.boxPoints(rect))
#     # 获取矩形中心点坐标，尺寸和角度
#     center,(w,h),angle=rect
#     if w > h:
#         angle=angle-90
#     # 旋转矩阵
#     M =cv2.getRotationMatrix2D(center,angle,1)
#     # 计算图像大小
#     (h,w)=mask.shape[:2]
#     # 考虑到旋转后的图像可能会超出原始图像，需要重新计算新的图像大小
#     cos =np.abs(M[0,0])
#     sin =np.abs(M[0,1])
#     # 转换宽高
#     nW=int(h*sin+w*cos)
#     nH=int(h*cos+w*sin)
#     # 计算平移量
#     tx = nW / 2 - center[0]
#     ty = nH / 2 - center[1]
#     # 更新旋转矩阵的平移部分
#     M[0, 2] += tx  # 原 M[0,2] 是 -center_x * cos + center_y * sin + center_x
#     M[1, 2] += ty  # 我们再加上居中所需的平移
#     # 旋转
#     rotate =cv2.warpAffine(mask,M,(nW,nH), flags=cv2.INTER_CUBIC)
#     # 更新转载后得矩形坐标
#     # rotated_box = cv2.boxPoints(cv2.minAreaRect(point))
#     rotated_box = cv2.transform(np.array([box]), M)[0]
#     rotated_box = np.intp(rotated_box)
#     # 测试
#     # cv2.polylines(rotate, [rotated_box], isClosed=True, color=255, thickness=2)
#     # 测试
#     # cv2.imwrite(f"mask.png", rotate)
#     return rotate



# 减少重复操作 速度快这是导致速度卡顿的最大问题
def aligning(mask, point):
    rect = cv2.minAreaRect(point)
    center, (w_rect, h_rect), angle = rect
    if w_rect > h_rect:
        angle -= 90
    M = cv2.getRotationMatrix2D(center, angle, 1)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    h, w = mask.shape[:2]
    nW = int(h * sin + w * cos)
    nH = int(h * cos + w * sin)
    M[0, 2] += (nW / 2 - center[0])
    M[1, 2] += (nH / 2 - center[1])
    rotate = cv2.warpAffine(mask, M, (nW, nH), flags=cv2.INTER_LINEAR)
    # cv2.imwrite(f"mask111.png", rotate)
    return rotate
