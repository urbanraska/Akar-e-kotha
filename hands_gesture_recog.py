import cv2
import mediapipe as mp
import pandas as pd  
import os
import numpy as np
from gtts import gTTS 
from playsound import playsound
import time
import asyncio
import pyttsx3

#speak section
# async def speak(temp):
#     language = 'en'
#     # await asyncio.sleep(1)
#     obj = gTTS(text=temp, lang=language, slow=False)
#     await asyncio.sleep(2)
#     obj.save("temp.mp3") 
#     # await asyncio.sleep(1)
#     playsound("temp.mp3")
#     # os.system("mpg321 temp.mp3")
#     print("Speaking.....")
#     await asyncio.sleep(1)
#     os.remove("temp.mp3")
#     # del(obj) 
# Initialise Text to speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 105)
engine.setProperty('voice', 1)
    
def speak(temp):
    engine.say(temp)
    engine.runAndWait()

def image_processed(hand_img):

    # Image processing
    # 1. Convert BGR to RGB
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    # 2. Flip the img in Y-axis
    img_flip = cv2.flip(img_rgb, 1)

    # accessing MediaPipe solutions
    mp_hands = mp.solutions.hands

    # Initialize Hands
    hands = mp_hands.Hands(static_image_mode=True,
    max_num_hands=2, min_detection_confidence=0.7)

    # Results
    output = hands.process(img_flip)

    hands.close()

    try:
        data = output.multi_hand_landmarks[0]
        #print(data)
        data = str(data)

        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_garbage = []

        for i in data:
            if i not in garbage:
                without_garbage.append(i)
                        
        clean = []

        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])
        return(clean)
    except:
        return(np.zeros([1,63], dtype=int)[0])

import pickle
# load model
with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)


import cv2 as cv
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
i = 0 
vTEXT=""   
while True:
    
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # frame = cv.flip(frame,1)
    data = image_processed(frame)
    
    # print(data.shape)
    data = np.array(data)
    y_pred = svm.predict(data.reshape(-1,63))
    # print(y_pred)
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # org
    org = (50, 100)
    
    # fontScale
    fontScale = 3
    
    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 5
    
    # Using cv2.putText() method
    frame = cv2.putText(frame, str(y_pred[0]), org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    # print(str(y_pred[0]))
    vTEXT=str(y_pred[0])
    
    # print(vTEXT)
    # imgarr=[]
    # imgarr = [vTEXT for i in range(3)]
    # print("speak",imgarr[0])
    c = cv2.waitKey(1) & 0xff
    cv.imshow('frame', frame)
    # if len(vTEXT) > 0:
    #     speak(vTEXT)
    if c == ord('q'):
        break 
    # if cv.waitKey(1) == ord('q'):
    #     break 
    # asyncio.run(speak(vTEXT))
    if len(vTEXT) > 0 and c == ord('z'):
        speak(vTEXT)
    # if c == ord('d'):
    #     os.remove("temp.mp3") 
    
cap.release()
cv.destroyAllWindows()