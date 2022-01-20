

import cv2
import numpy as np
from time import sleep
from time import time
import pafy


largura_min=80 #Largura minima do retangulo
altura_min=80 #Altura minima do retangulo

offset=6 #Erro permitido entre pixel  

pos_linha=550 #Posição da linha de contagem 

delay= 60 #FPS do vídeo

detec = []
carros= 0

	
def pega_centro(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

URL = "https://www.youtube.com/watch?v=spmC3OpP-JA" #URL to parse
play = pafy.new(URL).streams[-1] #'-1' means read the lowest quality of video.
assert play is not None # we want to make sure their is a input to read.
cap = cv2.VideoCapture(play.url) #create a opencv video stream.
# cap = cv2.VideoCapture('video.mp4')
subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()


start_time = time()
    

while True:
    ret , frame1 = cap.read()
    tempo = float(1/delay)
    sleep(tempo) 
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = subtracao.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
    dilatada = cv2.morphologyEx (dilatada, cv2. MORPH_CLOSE , kernel)
    contorno,h=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1, (25, pos_linha), (600, pos_linha), (255,127,0), 3) 
    for(i,c) in enumerate(contorno):
        (x,y,w,h) = cv2.boundingRect(c)
        validar_contorno = (w >= largura_min) and (h >= altura_min)
        if not validar_contorno:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        
        centro = pega_centro(x, y, w, h)
        detec.append(centro)
        cv2.circle(frame1, centro, 4, (0, 0,255), -1)

        for (x,y) in detec:
            # do stuff
            if time() - start_time > 60: # 60 secs
                print("There have passed "+str(carros)+" cars in the past 60 seconds")
                carros = 0
                start_time = time()   
            if y<(pos_linha+offset) and y>(pos_linha-offset) and x>(25+offset) and x<(600-offset):
                carros+=1
                cv2.line(frame1, (25, pos_linha), (600, pos_linha), (0,127,255), 3)  
                detec.remove((x,y))
                 
       
    cv2.putText(frame1, "VEHICLE COUNT : "+str(carros), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    # cv2.imshow("Video Original" , frame1)
    # cv2.imshow("Detectar",dilatada)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()