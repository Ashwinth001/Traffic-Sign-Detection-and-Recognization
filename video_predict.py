from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
import pandas as pd
import matplotlib.pyplot as plt

import random
SignName=pd.read_csv("Sign_Names.csv")

SignNames=pd.Series(SignName.SignName.values,index=SignName.ClassId).to_dict()

model = load_model("my_model")
lb = pickle.loads(open("label_bin.pickle", "rb").read())
# initialize the image mean for mean subtraction along with the
# predictions queue
Queue = deque(maxlen=10)

Video= 'Videos_dataset/VID_20161126_130348.mp4'
capture_video = cv2.VideoCapture(Video)

writer = None
(Height,Width) = (None, None)

# loop over frames from the video file stream


while True:
    (taken, frame) = capture_video.read()
    if not taken:
        break
    if Width is None or Height is None:
        (Height,Width) = frame.shape[:2]
        
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (32, 32)).astype('float32')
    frame = frame.reshape(-1,32,32).astype('float32')
    
    
    preds = model.predict(np.expand_dims(frame, axis=-1))[1]
    
      
    
    Queue.append(preds)
    results = np.array(Queue).mean(axis=0)
    i = np.argmax(results)
    
    label = lb.classes_[i]
      
    text = 'Traffic Sign : {}'.format(SignNames[label])
    cv2.putText(output, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 0), 5)
    
    if writer is None:
        # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        fps = capture_video.get(cv2.CAP_PROP_FPS)
        writer = cv2.VideoWriter("/Output/output_3.mp4", fourcc,25.0, (Width, Height))
        
    writer.write(output)
    cv2.imshow('In progress', output)
    key = cv2.waitKey(10) & 0xFF
    
    if key == ord('q'):
        break

        

print('Finalizing....')
writer.release()
capture_video.release()

cv2.destroyAllWindows()

for i in range (1,5):
    cv2.waitKey(1)