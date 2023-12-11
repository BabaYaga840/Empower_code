import numpy as np
import cv2

img = cv2.imread('Resources/origin1.png')
listy=[]
## 1. choose HSV
# rough hsv values in this image.
# grid(gray)=(0, 0, 174)
# border(black)=(0, 0, 15)
# back ground(white)=(0, 0, 254)

# hsv value including border and back ground
hsv_min = (0, 0, 0) # Lower end of the HSV range
hsv_max = (10, 10, 255) # Upper end of the HSV range

# Transform image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Threshold based on HSV values
color_thresh = cv2.inRange(hsv, hsv_min, hsv_max)

# Invert the image
invert = cv2.bitwise_not(color_thresh)

## If you need perform skeletonization, use skimage.
#import skimage
#from skimage.morphology import skeletonize
#color_thresh = skeletonize(skimage.img_as_float(color_thresh))
#color_thresh = color_thresh.astype('uint8') * 255

cv2.imwrite("lines.png", invert)

## 2. cv2.HoughLinesP
# Actually, parameter tuning will be necessary

minLineLength = 100
maxLineGap = 100
lines = cv2.HoughLinesP(color_thresh, 2, np.pi/180,70,minLineLength,maxLineGap)
# lines = [[[319 321 431 188]], ... ,[[ 83 283 195 399]]]

lines = [x.flatten() for x in lines]
# lines = [[319, 321, 431, 188], ... ,[ 83, 283, 195, 399]]

# sort by first element
lines = sorted(lines, key=lambda x : x[0])

# drow lines
for line in lines:
    x1,y1,x2,y2 = line
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)


## 3. Find the intersection of two lines
def cross_point(segment_p1p2, segment_p3p4):
    '''
    Intersection point of two lines
    p1p2:｛p1(a,b)、p2(c,d)｝
    p3p4:｛p3(e,f)、p4(g,h)｝
    '''
    a = segment_p1p2[0]
    b = segment_p1p2[1]
    c = segment_p1p2[2]
    d = segment_p1p2[3]
    e = segment_p3p4[0]
    f = segment_p3p4[1]
    g = segment_p3p4[2]
    h = segment_p3p4[3]
    print(a,b,c,d,e,f,g,h)

    dev = (d-b)*(g-e)-(c-a)*(h-f)
    if dev != 0:
        d1 = f*g-e*h
        d2 = b*c-a*d

        xp = (d1*(c-a)-d2*(g-e))/dev
        yp = (d1*(d-b)-d2*(h-f))/dev
        listy.append(yp)
        return (xp, yp)
    else:
        return (-1,-1)

# draw circles
for i in range(len(lines)-1):
    x, y = cross_point(lines[i], lines[i+1])
    cv2.circle(img,(int(x),int(y)), 5, (255,0,0), 2)

cv2.imwrite("out.png",img)

# [81, 281, 201, 406] and [81, 279, 198, 400] are overlapping
# [435, 184, 557, 214] and [451, 187, 558, 213] are overlapping
lines = [[81, 281, 201, 406],
         [81, 279, 198, 400],
         [193, 405, 313, 327],
         [312, 329, 437, 180],
         [435, 184, 557, 214],
         [451, 187, 558, 213]]

import math

def is_close(data1, data2, threshold=10):
    '''
    e.g.  data1=[x, y] , data1=[x, y, z]
    `threshold` is euclidean distance.
    calculate the distance between two points, and determine if they are in the neighborhood.
    '''
    return math.sqrt(sum((d1-d2)**2 for d1, d2 in zip(data1, data2))) < threshold

# sort by first element
lines = sorted(lines, key=lambda x : x[0])

chained_lines=[lines[0]]
index=0
for i in range(len(lines)):
    if i < index:
        continue
    end_of_line = lines[i][-2:]
    # generator which return the index of the chained elements
    y = (i for i, v in enumerate(lines) if is_close(end_of_line, v[:2]))
   # get the index of the first element
    chained_index = next(y, None)
    if chained_index != None and chained_index > index:
        index = chained_index
        chained_lines.append(lines[index])

print(chained_lines)



#img = cv2.imread('origin.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, bin_img = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)

minLineLength = 100
maxLineGap = 100
#lines = cv2.HoughLinesP(bin_img, 2, np.pi/180,70,minLineLength,maxLineGap)

#lines = [x.flatten() for x in chained_lines]

# drow lines
"""for line in chained_lines:
    x1,y1,x2,y2 = line
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)"""
print("-------------------------")
print(listy)
print(len(listy))
cv2.imwrite("out2.png",img)

