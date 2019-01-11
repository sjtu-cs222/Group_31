import cv2
import pandas as pd
import csv
import numpy as np
from numpy import array

img1 =cv2.imread("1.jpg")  #读取图片\
img2 =cv2.imread("2.jpg") 
img3 =cv2.imread("3.jpg") 
img4 =cv2.imread("4.jpg") 
img5 =cv2.imread("5.jpg") 
img6 =cv2.imread("6.jpg") 
img7 =cv2.imread("7.jpg") 
img8 =cv2.imread("8.jpg") 
img9 =cv2.imread("9.jpg") 
img10 =cv2.imread("10.jpg") 
 

img=np.array([img1,img2,img3,img4,img5,img6,img7,img8,img9,img10])
print (type(img[0]))


csvFilex = open("trainx.csv", "a", newline='') 
csvFiley = open("trainy.csv", "a", newline='')
writerx = csv.writer(csvFilex)           
writery = csv.writer(csvFiley)
writerx.writerow(["R","G","B"])
writery.writerow(["R","G","B"])
 
for i in range(len(img)):
    for j in img[i]:   
        writerx.writerows(j)
    for k in img[i+1]:
        writery.writerows(k)
    print ('done')
	
 

 
    
		

 
 
   	 
 
 
        
    
 	
 
 
 
 
 
 