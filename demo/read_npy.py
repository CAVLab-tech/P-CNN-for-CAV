import numpy as np

conv1_2= np.load('conv1_2.npy')
conv1_2=conv1_2[0][0]
conv1_2_1= np.load('conv1_2_1.npy')
conv1_2_1=conv1_2_1[0][0]
conv1_2_2= np.load('conv1_2_2.npy')
conv1_2_2=conv1_2_2[0][0]
conv1_2_error= np.load('conv1_2_error.npy')
conv1_2_error=conv1_2_error[0][0]

relu1_2= np.load('relu1_2.npy')
relu1_2=relu1_2[0][0]
relu1_2_1= np.load('relu1_2_1.npy')
relu1_2_1=relu1_2_1[0][0]
relu1_2_2= np.load('relu1_2_2.npy')
relu1_2_2=relu1_2_2[0][0]
relu1_2_error= np.load('relu1_2_error.npy')
relu1_2_error=relu1_2_error[0][0]

pool1= np.load('pool1.npy')
pool1=pool1[0][0]
pool1_1= np.load('pool1_1.npy')
pool1_1=pool1_1[0][0]
pool1_2= np.load('pool1_2.npy')
pool1_2=pool1_2[0][0]
pool1_error= np.load('pool1_error.npy')
pool1_error=pool1_error[0][0]

fc7= np.load('fc7.npy')
fc7_1= np.load('fc7_1.npy')
fc7_2= np.load('fc7_2.npy')
fc7_error= np.load('fc7_error.npy')

score= np.load('score.npy')
score1= np.load('score1.npy')
score2= np.load('score2.npy')
score_error= np.load('score_error.npy')

relu5_3= np.load('relu5_3.npy') #1,512,14,14
relu5_3 = relu5_3.transpose(2, 3, 0, 1)
relu5_3=relu5_3[0][0]

