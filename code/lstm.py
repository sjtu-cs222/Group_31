import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import numpy as np
import csv
import os
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.layers.core import Activation
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import cv2
from numpy import array

os.chdir("E:\工作区\算法设计\image predict")
f1=open('trainx.csv') 
df1=pd.read_csv(f1)      
l1=df1.iloc[:].values
cn=0
def get_x(img_num,color_num,csv_data):
    color=np.zeros((120*120*img_num,9))
    num=0
    for img in range (img_num):
        for i in range(120):
            for j in range(120):
                color[num][0]=csv_data[img*128*128+i*128+j][color_num]
                color[num][1]=csv_data[img*128*128+i*128+j+1][color_num]
                color[num][2]=csv_data[img*128*128+i*128+j+2][color_num]		
                color[num][3]=csv_data[img*128*128+(i+1)*128+j][color_num]
                color[num][4]=csv_data[img*128*128+(i+1)*128+j+1][color_num]
                color[num][5]=csv_data[img*128*128+(i+1)*128+j+2][color_num]	
                color[num][6]=csv_data[img*128*128+(i+2)*128+j][color_num]
                color[num][7]=csv_data[img*128*128+(i+2)*128+j+1][color_num]
                color[num][8]=csv_data[img*128*128+(i+2)*128+j+2][color_num]
                num+=1
    return color
f2=open('trainy.csv') 
df2=pd.read_csv(f2)      
l2=df2.iloc[:].values
def get_y(img_num,color_num,csv_data):
   color=np.zeros((120*120*img_num,1))
   num=0
   for img in range (img_num):
       for i in range(120):
           for j in range(120):
               color[num]=csv_data[img*128*128+(i+1)*128+j+1][color_num]
               num+=1
   return color

ft = open("test.csv")
dft = pd.read_csv(ft)
testData = dft.iloc[:].values
ft.close()
cn=0
for i in range(3):
	train_in=get_x(9,cn,l1)
	#print (x_train)
	train_out=get_y(9,cn,l2)
	#print (y_train)
	cut=int(len(train_in)*0.99)
	x_train=np.array(train_in[:cut],ndmin=2)
	y_train=np.array(train_out[:cut])
	x_val=np.array(train_in[cut:],ndmin=2)
	y_val=np.array(train_out[cut:])
	
	x_test = get_x(1,cn,testData)

	model =Sequential()
	model.add(LSTM(128,input_shape=(x_train.shape[1],x_train.shape[2]),return_sequences=True))
	model.add(Dropout(0.5))
	model.add(LSTM(128,return_sequences=True))
	model.add(LSTM(128,return_sequences=False))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam', metrics=['mse'])
	callback=EarlyStopping(monitor="loss",patience=10,verbose=1,mode='auto')
	model.fit(x_train, y_train, epochs=4,validation_data=(x_val,y_val), batch_size=128, callbacks=[callback],shuffle=True)

	res=model.predict(x_test)
	res.to_csv("output"+str(cn)+".csv")
	cn+=1

