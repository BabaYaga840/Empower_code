import numpy as np
import cv2
from cv2 import aruco
thicc=80
j=0
listy=[]
size=[]
vol=600
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
    success, img = cap.read()
    h = img.shape[0]
    w = img.shape[1]
    print(h)
    print(w)
    img = img[0:h , w // 4:3 * w // 4]
    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
while True:


    success, img=cap.read()
    h = img.shape[0]
    w = img.shape[1]
    img = img[0:h , w // 4:3 * w // 4]
    h = img.shape[0]
    w = img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    """aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
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
            img=img[int(corners[0][0][0][0]):int(corners[1][0][1][0]),int(corners[0][0][0][1]):int(corners[1][0][1][1])]"""
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
    a=np.zeros((w//thicc+1),int)
    b=np.zeros((h//thicc+1),int)
    for i in range(0,l):
        if listy[i]>10 and (listy[i]<w-10 and (size[i]>10 and size[i]<h-10)) :
            a[listy[i]//thicc]+=1
            b[size[i]//thicc]+=1
    maxy=np.argmax(a)
    maxx=np.argmax(b)
    a[maxy]=0
    b[maxx]=0
    maxy2=np.argmax(a)
    maxx2=np.argmax(b)
    a[maxy2] = 0
    b[maxx2] = 0
    maxy3 = np.argmax(a)
    maxx3 = np.argmax(b)
    print(maxx*30,maxx2*30,maxx3*30)
    xavg1=xavg2=xavg3=0
    n1=n2=n3=0
    for m in size:
        if m//thicc==maxx:
            xavg1+=m
            n1+=1
        if m//thicc==maxx2:
            xavg2+=m
            n2+=1
        if m//thicc==maxx3:
            xavg3+=m
            n3+=1
    xavg1=xavg1/n1
    xavg2=xavg2/n2
    xavg3=xavg3/n3
    if xavg1<xavg2 and xavg1<xavg3:
        volume=(xavg2-xavg1+10)/(xavg3-xavg1)
    if xavg2<xavg1 and xavg2<xavg3:
        volume=(xavg1-xavg2+10)/(xavg3-xavg2)
    if xavg3<xavg2 and xavg3<xavg1:
        volume=(xavg2-xavg3+10)/(xavg1-xavg3)
    if volume>1:
        volume=1/volume
    volume*=vol
    print(600-volume)

    cv2.waitKey(0)
