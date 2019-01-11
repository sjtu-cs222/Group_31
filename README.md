The projects require following libraries:
Tensorflow 1.2.1 
Python 1.6.2
Pandas 0.23.4
Numpy 1.12.1
Xgboost 0.6
keras 2.1.2
Sklearn 0.19.0
Steps to run the project:
1. Run getrain.py to transfer input image into .csv file. The gettest.py is actually in same method but created for convenience.
2. Ensure the file name is correct: trainx.csv, trainy.csv (1 frame after trainx) and test.csv
3. Run xgb.py and there should be 3 output files. Put them together in one file named "output.csv"
4. Run writepicture.py to transfer "output.csv" into a picture named "output.jpg"
P.s.: dataset is included (cloud, flag and plate)
