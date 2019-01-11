import cv2
import numpy as np
from numpy import array
import pandas as pd
import csv
import os
os.chdir("E:\工作区\算法设计\image predict")

f=open('output.csv') 
df=pd.read_csv(f)      
l=df.iloc[:].values
#print (type(l))

img_test=np.zeros((178,318,3))
for i in range(178):
    for j in range(318):
        for k in range(3):
            img_test[i][j][k]=l[i*318+j][k]
             
cv2.imwrite("output.jpg",img_test)
