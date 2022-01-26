import numpy as np
import cv2
from time import sleep
from time import time
import pafy
import requests

min_contour_width=90  #40
min_contour_height=90  #40
offset=6       #10
line_height=550 #550
matches =[]
cars=0
def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1
    return cx,cy
    #return (cx, cy)

URL = "https://www.youtube.com/watch?v=y7QiNgui5Tg" #URL to parse
play = pafy.new(URL).streams[-1] #'-1' means read the lowest quality of video.
assert play is not None # we want to make sure their is a input to read.
cap = cv2.VideoCapture(play.url) #create a opencv video stream.
# cap = cv2.VideoCapture('video.mp4')   
#cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('video.mp4')
#cap = cv2.VideoCapture('Relaxing highway traffic.mp4')





if cap.isOpened():
    ret,frame1 = cap.read()
else:
    ret = False
ret,frame1 = cap.read()
ret,frame2 = cap.read()


start_time = time()
print('application start')

while ret:
    d = cv2.absdiff(frame1,frame2)
    grey = cv2.cvtColor(d,cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(grey,(5,5),0)
    blur = cv2.GaussianBlur(grey,(5,5),0)
    #ret , th = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    ret , th = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    dilated = cv2.dilate(th,np.ones((3,3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

        # Fill any small holes
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel) 
    contours,h = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for(i,c) in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect(c)
        contour_valid = (w >= min_contour_width) and (
            h >= min_contour_height)

        if not contour_valid:
            continue
        cv2.rectangle(frame1,(x-10,y-10),(x+w+10,y+h+10),(255,0,0),2)
        
        cv2.line(frame1, (1100, 750), (1100, 300), (255,127,0), 3) 
        centroid = get_centroid(x, y, w, h)
        matches.append(centroid)
        cv2.circle(frame1,centroid, 5, (0,255,0), -1)
        cx,cy= get_centroid(x, y, w, h)
        for (x,y) in matches:
            if time() - start_time > 60: # 60 secs
                # tijd = datetime.now()
                # print(tijd.hour)
                # print(tijd.minute+1)
                # print("There have passed "+str(cars)+" cars in the past 60 seconds")
                post_url = "https://project40-tm.herokuapp.com/stats/car"
                myobj = {'amount': cars}

                requests.post(post_url, json = myobj)
                cars = 0
                start_time = time()  
            if y<(750+offset) and y>(300-offset) and x>(1100-offset) and x<(1100+offset):
                cars=cars+1
                matches.remove((x,y))
                
    # cv2.putText(frame1, "Total Cars Detected: " + str(cars), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #                 (0, 170, 0), 2)

    # cv2.putText(frame1, "MechatronicsLAB.net", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #                 (255, 170, 0), 2)
    
    
    
    # cv2.drawContours(frame1,contours,-1,(0,0,255),2)


    # cv2.imshow("Original" , frame1)
    # cv2.imshow("Difference" , th)
    # if cv2.waitKey(1) == 27:
    #     break
    frame1 = frame2
    ret , frame2 = cap.read()
#print(matches)    
# cv2.destroyAllWindows()
# cap.release()