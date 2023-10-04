# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 02:57:22 2023

@author: 3ndalib
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
Image("ID2.jpg")
def prepro(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mask = np.zeros((gray.shape),np.uint8)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))

    close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel1)
    div = np.float32(gray)/(close)
    res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
    res2 = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)
    plt.imshow(res2)
    return res2, res, mask
m = cv2.imread("ID2.jpg")
ref = cv2.imread("ID2.jpg")
print(type(ref))
ref_g = cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)    
mo, mg, mask = prepro(m)

plt.imshow(mask)
plt.imshow(mg)
plt.imshow(ref_g)
thresh = cv2.adaptiveThreshold(mg,255,0,1,19,2)
contour,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
plt.imshow(thresh)
max_area = 0
best_cnt = None
for cnt in contour:
    area = cv2.contourArea(cnt)
    if area > 1000:
        if area > max_area:
            max_area = area
            best_cnt = cnt

t1 = cv2.drawContours(mask,[best_cnt],0,255,-1)

t2 = cv2.drawContours(mask,[best_cnt],0,0,2)


res = cv2.bitwise_and(mg,mask)
plt.imshow(mask)
plt.imshow(res)
uwu = mo.copy()
for i in best_cnt:
    #print(i[0])
    cv2.circle(uwu,(i[0][0], i[0][1]), 5, (0,255,0), -1)
    
plt.imshow(uwu)

canny = cv2.Canny(mask, 120, 255, 1)
plt.imshow(canny)
corners = cv2.goodFeaturesToTrack(canny,4,0.5,50)

uwu2 = mo.copy()
for corner in corners:
    x,y = corner.ravel()
    cv2.circle(uwu2,(x,y),5,(36,255,12),-1)
    cv2.putText(uwu2,'{},{}'.format(int(x),int(y)),(x,y),2,2,(255,0,0))
    print(x,y)

plt.imshow(uwu2)

pA = corners[0][0]
pB = corners[1][0]
pC = corners[2][0]
pD = corners[3][0]

m.shape
mH = m.shape[0]
mW = m.shape[1]

Image("ID2.jpg")


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = (nodes - node)**2
    print(dist_2.reshape(4,2))
    n = np.sum(dist_2.astype("int"),axis=-1)

    return np.argmin(n)


pA = corners[closest_node([0,0],corners)]
pB = corners[closest_node([0,mH-1],corners)]
pC = corners[closest_node([mW-1,mH-1],corners)]
pD = corners[closest_node([mW-1,0],corners)]

H = W = 252

output_pts = np.float32([[0, 0],
                        [0, H - 1],
                        [W - 1, H - 1],
                        [W - 1, 0]])

input_pts = np.float32([pA, pB, pC, pD])
M = cv2.getPerspectiveTransform(input_pts,output_pts)

out = cv2.warpPerspective(mo,M,(W, H),flags=cv2.INTER_LINEAR)


plt.imshow(out)
