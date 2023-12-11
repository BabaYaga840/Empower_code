import numpy as np
import cv2
from cv2 import aruco

j=0
listy=[]
size=[]
def getContours(img,img1):

    global j
    global listy
    global size
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    listy = []
    size = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 150:  # remove noise


            approx = cv2.approxPolyDP(cnt, 0.09 * cv2.arcLength(cnt, True), False)  # extract points from contour
            n = approx.ravel()
            i = 0
            #size.append(len(n))
            x=0
            for pt in n:
                if (i % 2 == 0):
                    listy.append(pt)
                else:
                    size.append(pt)
                i = i + 1
            """listy.append(maxy)
            size.append(x1)
            listy.append(miny)
            size.append(x2)"""


            #print(area)
            j+=1
            cv2.drawContours(img1, cnt, 2 , (0,255,0), 3)
            cv2.imshow("canny", img1)
            cv2.imshow("f1", img)

cap=cv2.VideoCapture(0)
if not cap.isOpened():
    print("cap not open")
    exit()

while True:


    success, img=cap.read()
    h = img.shape[0]
    w = img.shape[1]
    img=img[h//4:3*h//4,w//4:3*w//4]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    l=ids
    print(corners)
    if(str(type(ids))!= "<class 'NoneType'>"):
        print(type(corners[0][0][0]))
        print("hgd")
        if(len(ids)>1):
            print("corners", corners[0][0][0][0])
            #print(corners[0,0][0])
            img=img[int(corners[0][0][0][0]):int(corners[1][0][1][0]),int(corners[0][0][0][1]):int(corners[1][0][1][1])]
    print(h)
    print(w)
    if not success:
        print("unable to get vid")
        break
    #cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #cv2.imshow("i", img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avgv=np.average(hsv[:,:,2])
    hsv[hsv[:,:]<avgv]=0
    hsv[hsv[:,:]>=avgv]=255
    hsv[:,:,1]=0
    hsv[:,:,0]=0
    # Transform image to HSV color space


    # Threshold based on HSV values
    #color_thresh = cv2.inRange(hsv, hsv_min, hsv_max)
    #img1=hsv
    # Invert the image
    #img1 = cv2.bitwise_not(color_thresh)
    #cv2.imshow("i", img)
    img1=cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2,2), np.uint8)
    #img = cv2.erode(img, kernel, iterations=10)
    #img1 = cv2.erode(img1, kernel, iterations=8)
    #img1=cv2.Canny(img1,50,150)
    getContours(img1,img)

    l=len(listy)
    l1=len(size)
    for a in range(0,l):
        cv2.circle(img, (listy[a],size[a]),radius=5,color=(0,0,255),thickness=-1)
    edges = cv2.Canny(img, 100, 200)
    cv2.imshow("can", edges)
    cv2.imshow("f", img)
    cv2.imshow("img", img1)
    print("----------------")
    print(listy)
    print(l)
    print(size)
    print(l1)
    cv2.waitKey(100)


#def calculate(listy):

"""gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

import cv2
import numpy as np

#img = cv2.imread('sofsk.png', 0)
size = np.size(img)
skel = np.zeros(img.shape, np.uint8)

ret, img = cv2.threshold(img, 127, 255, 0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
done = False

while (not done):
    eroded = cv2.erode(img, element)
    temp = cv2.dilate(eroded, element)
    temp = cv2.subtract(img, temp)
    cv2.imshow("skel", skel)
    cv2.imshow("temp", temp)
    skel = cv2.bitwise_or(skel, temp)
    img = eroded.copy()

    """#zeros = size - cv2.countNonZero(img)
    #if zeros == size:
     #   done = True"""

"""cv2.imshow("skel", skel)
print("UwU")"""

cv2.waitKey(0)

