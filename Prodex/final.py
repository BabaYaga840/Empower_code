import numpy as np
import cv2

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
        if area > 50:  # remove noise


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

img=cv2.imread("Resources/fin1.png")
h = img.shape[0]
w = img.shape[1]
print(h)
print(w)
#img = img[0:h , w // 4:3 * w // 4]
cv2.imshow("image", img)

h = img.shape[0]
w = img.shape[1]
#img = img[h // 4:3 * h // 4, w // 4:3 * w // 4]
h = img.shape[0]
w = img.shape[1]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(h)
print(w)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
avgv=np.average(hsv[:,:,2])
hsv[hsv[:,:]<avgv]=0
hsv[hsv[:,:]>=avgv]=255
hsv[:,:,1]=0
hsv[:,:,0]=0

img1=cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
#kernel = np.ones((2,2), np.uint8)

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
print(maxx*thicc,maxx2*thicc,maxx3*thicc)
xavg1=xavg2=xavg3=0
n1=n2=n3=0
#if(n3<(n1+n2)/2-5):
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
