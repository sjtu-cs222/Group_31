import cv2
import numpy as np
from numpy import array
import pandas as pd
import csv
import os
os.chdir("E:\工作区\算法设计\image predict")
f=open('trainy.csv') 
df=pd.read_csv(f)      
l=df.iloc[:].values
 
def get_y(img_num,color_num,csv_data):
   color=np.zeros((178*318*img_num,1))
   num=0
   for img in range (img_num):
       for i in range(178):
           for j in range(318):
               color[num][0]=csv_data[img*320*180+(i+1)*180+j+1][color_num]
               num+=1
   return color
print(get_y(9,0,l))