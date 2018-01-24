import cv2
import numpy as np

def getLandmarks(corners):
    holes=[]
    for i in range(0,len(corners)):
        for j in range(i+1,len(corners)):
            x1,y1=corners[i].ravel()
            x2,y2=corners[j].ravel()
            if abs(x1-x2)<=30 and abs(y1-y2)<=30:
                holes.append((int((x1+x2)/2),int((y1+y2)/2)))
    return holes

# lodes in img

img = cv2.imread('img.png', cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

corners = cv2.goodFeaturesToTrack(img_gray, 200, 0.05, 10)

holes=getLandmarks(corners)
print len(holes)
for corner in holes:
    cv2.circle(img, (corner), 7, (255,255,0), -1)

cv2.imshow('img',img)
cv2.waitKey(0)
