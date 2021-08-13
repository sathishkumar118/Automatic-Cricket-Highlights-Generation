import cv2
import numpy as np
import pytesseract
from PIL import Image
#from __future__ import print_function
#from builtins import input
import argparse
import time
start = time.time()

vidcap = cv2.VideoCapture("C:/FYP/Videos/6_BBL_Trim.mp4 ")
vidcap.set(cv2.CAP_PROP_POS_MSEC,313*1000)      # just cue to 0th sec. position
# 306-6 212-4 240 230-6 121-8 16-4 195-4
success,myimg=vidcap.read()
cv2.imwrite("frame20sec.jpg",myimg)
myimg=cv2.imread("frame20sec.jpg")
myimg = cv2.cvtColor(myimg, cv2.COLOR_BGR2GRAY)
#kernel = np.ones((1, 1), np.uint8)
#myimg = cv2.dilate(myimg, kernel, iterations=1)
#new_image=myimg
new_image = np.zeros(myimg.shape, myimg.dtype)
new_image = cv2.convertScaleAbs(myimg, alpha=0.90, beta=50)
#new_image = cv2.convertScaleAbs(myimg, alpha=2.0, beta=10)


#kernel = np.ones((1, 1), np.uint8)
#myimg = cv2.dilate(myimg, kernel, iterations=1)
#myimg = cv2.erode(myimg, kernel, iterations=1)
cropped=new_image[630:660, 265:346]
cv2.imwrite("mycrop.jpg",cropped)

#cv2.imshow("cropped",cropped)
#cv2.waitKey(0)
result = pytesseract.image_to_string(cropped)
result=result.replace(" ","")
print(result)
end = time.time()
dur = end-start
if dur<60:
    print("aggregation Time:",dur,"seconds")
elif dur>60 and dur<3600:
    dur=dur/60
    print("aggregation Time:",dur,"minutes")
else:
    dur=dur/(60*60)
    print("aggregation Time:",dur,"hours")
