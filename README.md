# Python code

Open a new folder and place a new python file in this folder. This will be your project folder. Open your project in your favourite code editor and navigate to your project folder.

## Import Libraries

```
import numpy as np
import cv2
from time import time
import pafy
import requests
```

requests -> to send data using post

numpy and cv2 -> to analyze data from a video

pafy -> to make opencv work with youtube-dl so you can use youtube videos

## Define variables

```
#tweak these settings to get more accurate predictions
min_contour_width=100  #original = 40 
min_contour_height=100  # original = 40
offset=5       # original = 10


# define variables and functions
matches =[]
start_time = time()
seconds = 600
cars=0
def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1
    return cx,cy
    #return (cx, cy)
```

## Choose the input

Decide what your input will be. To change the Youtube video, just change the URL variable. In this example we use a youtube livestream. but it also can be:

- name of video file (eg. video.avi)
- image (eg. img_02.jpg)
- URL of video stream (eg. protocol://host:port/script_name script_params|auth)
- a camera attached to the device (eg. 0 or 1)

```
# cap = cv2.VideoCapture(0) will open the webcam
URL = "https://www.youtube.com/watch?v=y7QiNgui5Tg" #URL to parse
play = pafy.new(URL).streams[-1] #'-1' means read the lowest quality of video.
assert play is not None # we want to make sure their is a input to read.
cap = cv2.VideoCapture(play.url) #create a opencv video stream.


if cap.isOpened():
    ret,frame1 = cap.read()
else:
    ret = False
ret,frame1 = cap.read()
ret,frame2 = cap.read()
```
## Detect movement

With these lines of code we can detect mouvement on our video. Around the moving objects a rectangle will be drawn so the program can track its position. If the object (car) passes a certain x and y value, increase the cars counter. Send after some time a post request with the amount of cars passed and set the counter back to 0.

```
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

            # do a post request and send the amount of cars passed in the last .. seconds
            if time() - start_time > seconds:
                post_url = "https://project40-tm.herokuapp.com/stats/car"
                myobj = {'amount': cars}
                requests.post(post_url, json = myobj)
                cars = 0
                start_time = time()

            #if car passed a certain line, increase the counter of cars with one    
            if y<(750+offset) and y>(300-offset) and x>(1100-offset) and x<(1100+offset):
                cars=cars+1
                matches.remove((x,y))

# uncomment the lines below to see what the code does if you run it locally

    # cv2.drawContours(frame1,contours,-1,(0,0,255),2)
    # cv2.imshow("Original" , frame1)
    # cv2.imshow("Difference" , th)
    # if cv2.waitKey(1) == 27:
    #     break
    frame1 = frame2
    ret , frame2 = cap.read()
# print(matches)    
# cv2.destroyAllWindows()
# cap.release()
```

You are now ready to run your code. Test it locally and see if it works! Try to change the variables to make the predictions even more accurate.
