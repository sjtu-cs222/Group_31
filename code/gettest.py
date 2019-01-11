import cv2
import pandas as pd
import csv
import numpy as np
from numpy import array
import os
os.chdir("E:\工作区\算法设计\image predict")
 
img10 =cv2.imread("c1.png") 
csvFile= open("test1.csv", "a", newline='') 
 
writer = csv.writer(csvFile)           
 
writer.writerow(["R","G","B"])
for j in img10:   
	writer.writerows(j)
 
	
 

 
    
		

 
 
   	 
 
 
        
    
 	
 
 
 
 
 
 