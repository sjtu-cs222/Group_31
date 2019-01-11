import pandas as pd
import numpy as np
import xgboost as xgb
import cv2
from numpy import array
import csv
import os
os.chdir("E:\工作区\算法设计\image predict")

f1=open('trainx.csv') 
df1=pd.read_csv(f1)      
l1=df1.iloc[:].values
cn=0
def get_x(img_num,color_num,csv_data):
    color=np.zeros((178*318*img_num,9))
    num=0
    for img in range (img_num):
        for i in range(178):
            for j in range(318):
                color[num][0]=csv_data[img*320*180+i*320+j][color_num]
                color[num][1]=csv_data[img*320*180+i*320+j+1][color_num]
                color[num][2]=csv_data[img*320*180+i*320+j+2][color_num]		
                color[num][3]=csv_data[img*320*180+(i+1)*320+j][color_num]
                color[num][4]=csv_data[img*320*180+(i+1)*320+j+1][color_num]
                color[num][5]=csv_data[img*320*180+(i+1)*320+j+2][color_num]	
                color[num][6]=csv_data[img*320*180+(i+2)*320+j][color_num]
                color[num][7]=csv_data[img*320*180+(i+2)*320+j+1][color_num]
                color[num][8]=csv_data[img*320*180+(i+2)*320+j+2][color_num]
                num+=1
    return color
f2=open('trainy.csv') 
df2=pd.read_csv(f2)      
l2=df2.iloc[:].values
def get_y(img_num,color_num,csv_data):
   color=np.zeros((178*318*img_num,1))
   num=0
   for img in range (img_num):
       for i in range(178):
           for j in range(318):
               color[num][0]=csv_data[img*320*180+(i+1)*320+j+1][color_num]
               num+=1
   return color

ft = open("test.csv")
dft = pd.read_csv(ft)
testData = dft.iloc[:].values
ft.close()
for i in range(3):
    x_train=get_x(9,cn,l1)
    #print (x_train)
    y_train=get_y(9,cn,l2)
    #print (y_train)
    data_train = xgb.DMatrix(x_train, y_train)

    param = {'max_depth': 6, 'eta': 0.5, 'objective': 'reg:linear'}
    n_round = 3
    watchlist = [(data_train, 'train')]
    booster = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)

    x_test = get_x(1,cn,testData)
    #print(x_test)
    data_test = xgb.DMatrix(x_test)
    y_pre = booster.predict(data_test)
    y_pre = [list(y_pre)]
    y_pre = np.array(y_pre, ndmin = 2)
    y_pre = np.transpose(y_pre)
    c=[["R"],["G"],["B"]]
    pre= pd.DataFrame(columns = c[cn], data = y_pre)
    pre.to_csv("output"+str(cn)+".csv")
    cn+=1