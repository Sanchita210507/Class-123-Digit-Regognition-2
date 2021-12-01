import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# PIL = Python Imaging Library
from PIL import Image

import PIL.ImageOps
import os, ssl

#Setting an HTTPS Context to fetch data from OpenML
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context


X, y = fetch_openml('mnist_784',version = 1, return_X_y=True)
X=np.array(X)
print(pd.Series(y).value_counts())
classes = ['0','1','2','3','4','5','6','7','8','9']
nclasses = len(classes)

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=9,train_size = 7500, test_size = 2500)
# scaling the features
x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0
clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(x_train_scaled, y_train)
ypred = clf.predict(x_test_scaled)
accuracy = accuracy_score(y_test,ypred)
print("Accuracy is:",accuracy)

cap = cv2.VideoCapture(0)

while(True):
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        upperleft = (int(width/2-56),int(height/2-56))
        bottomright = (int(width/2+56),int(height/2+56)) 
        cv2.rectangle(gray,upperleft,bottomright,(0,255,0),2)
        roi = gray[upperleft[1]:bottomright[1],upperleft[0]:bottomright[0]]
        # converting cv2 image to PIL format.
        imPIL = Image.fromarray(roi)
        imgbw = imPIL.convert('L')
        imgbwresized = imgbw.resize((28,28),Image.ANTIALIAS)
        imgbwinverted = PIL.ImageOps.invert(imgbwresized)
        pixelFilter = 20
        # Converting to scalar quantity
        minPixel = np.percentile(imgbwinverted, pixelFilter)
        # limting the values between 0 and 255
        imgbwscaled = np.clip(imgbwinverted-minPixel,0,255)
        maxPixel = np.max(imgbwinverted)
        imgbwscaled = np.asarray(imgbwscaled)/maxPixel
        testSample = np.array(imgbwscaled).reshape(1,784)
        testPred = clf.predict(testSample)
        print("Predicted Value is: ",testPred)
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()
