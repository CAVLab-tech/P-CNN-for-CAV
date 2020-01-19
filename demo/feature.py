import numpy as np
import pylab
import pickle as p
import time

import sys
caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')

# caffe_root = '/home/cqx/caffe'
# sys.path.append('/home/cqx/caffe/python')

import caffe

import feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9
import feature10, feature11, feature12, feature13, feature14, feature15, feature16, feature17, feature18
import feature19, feature20, feature21, feature22, feature23, feature24, feature25, feature26, feature27
import feature28, feature29, feature30, feature31, feature32, feature33, feature34, feature35, feature36

import feature1_1, feature1_2, feature2_1, feature2_2, feature3_1, feature3_2, feature4_1, feature4_2
import feature5_1, feature5_2, feature6_1, feature6_2, feature7_1, feature7_2, feature8_1, feature8_2
import feature9_1, feature9_2, feature10_1, feature10_2, feature11_1, feature11_2, feature12_1, feature12_2
import feature13_1, feature13_2, feature14_1, feature14_2, feature15_1, feature15_2, feature16_1, feature16_2
import feature17_1, feature17_2, feature18_1, feature18_2, feature19_1, feature19_2, feature20_1, feature20_2
import feature21_1, feature21_2, feature22_1, feature22_2, feature23_1, feature23_2, feature24_1, feature24_2
import feature25_1, feature25_2, feature26_1, feature26_2, feature27_1, feature27_2, feature28_1, feature28_2
import feature29_1, feature29_2, feature30_1, feature30_2, feature31_1, feature31_2, feature32_1, feature32_2
import feature33_1, feature33_2, feature34_1, feature34_2, feature35_1, feature35_2, feature36_1, feature36_2

caffe.set_mode_gpu()
#caffe.set_mode_cpu()

import cv2
import os

t1 = time.time()

image_in = cv2.imread('rgb_data/testing/3.png')  #读取图片像素值
crop_size = (224, 224)
image = cv2.resize(image_in, crop_size, interpolation = cv2.INTER_AREA)     #转换成224×224×3的维度

image=image.reshape(224,224,-1)
image1 = np.random.randint(2**7, size=(224, 224, 3),dtype=np.int8)    #子图像1
image2 = image - image1         ##子图像2

image1 = image1.transpose(2, 0, 1)   #转置
image2 = image2.transpose(2, 0, 1)
image = image.transpose(2, 0, 1)

np.save('image_in.npy',image_in)
np.save('image.npy',image)
np.save('image1.npy',image1)
np.save('image2.npy',image2)


print('..................image......................')
print(image-(image1+image2))
print('..................end......................')


conv1_1 = feature1.f1(image)     #image
conv1_1_1 = feature1_1.f1_1(image1)   #image1
conv1_1_2 = feature1_2.f1_2(image2)   #image2
conv1_1_dim=np.shape(conv1_1_2)    #same_dimension
conv1_1_b= np.load('conv1_1_b.npy')   #load_bias
conv1_1_b=conv1_1_b.reshape(conv1_1_dim[0],conv1_1_dim[1],1,1)   #two_dim to four_dim
conv1_1_2=conv1_1_2-conv1_1_b    #sub_bias
conv1_1_error=conv1_1-(conv1_1_1+conv1_1_2)
np.save('conv1_1.npy',conv1_1)
np.save('conv1_1_1.npy',conv1_1_1)
np.save('conv1_1_2.npy',conv1_1_2)
np.save('conv1_1_error.npy',conv1_1_error)

relu1_1 = feature2.f2(conv1_1)
relu1_1_1 = feature2_1.f2_1(conv1_1_1, conv1_1_2)
relu1_1_2 = feature2_2.f2_2(conv1_1_1, conv1_1_2)
relu1_1_error=relu1_1-(relu1_1_1+relu1_1_2)
np.save('relu1_1.npy',relu1_1)
np.save('relu1_1_1.npy',relu1_1_1)
np.save('relu1_1_2.npy',relu1_1_2)
np.save('relu1_1_error.npy',relu1_1_error)

conv1_2 = feature3.f3(relu1_1)
conv1_2_1 = feature3_1.f3_1(relu1_1_1)
conv1_2_2 = feature3_2.f3_2(relu1_1_2)
conv1_2_dim=np.shape(conv1_2_2)    #same_dimension
conv1_2_b= np.load('conv1_2_b.npy')   #load_bias
conv1_2_b=conv1_2_b.reshape(conv1_2_dim[0],conv1_2_dim[1],1,1)   #two_dim to four_dim
conv1_2_2=conv1_2_2-conv1_2_b    #sub_bias
conv1_2_error=conv1_2-(conv1_2_1+conv1_2_2)
np.save('conv1_2.npy',conv1_2)
np.save('conv1_2_1.npy',conv1_2_1)
np.save('conv1_2_2.npy',conv1_2_2)
np.save('conv1_2_error.npy',conv1_2_error)

relu1_2 = feature4.f4(conv1_2)
relu1_2_1 = feature4_1.f4_1(conv1_2_1, conv1_2_2)
relu1_2_2 = feature4_2.f4_2(conv1_2_1, conv1_2_2)
relu1_2_error=relu1_2-(relu1_2_1+relu1_2_2)
np.save('relu1_2.npy',relu1_2)
np.save('relu1_2_1.npy',relu1_2_1)
np.save('relu1_2_2.npy',relu1_2_2)
np.save('relu1_2_error.npy',relu1_2_error)

pool1 = feature5.f5(relu1_2)
pool1_1 = feature5_1.f5_1(relu1_2_1,relu1_2_2)
pool1_2 = feature5_2.f5_2(relu1_2_1,relu1_2_2)
pool1_error=pool1 -(pool1_1+pool1_2)
np.save('pool1.npy',pool1)
np.save('pool1_1.npy',pool1_1)
np.save('pool1_2.npy',pool1_2)
np.save('pool1_error.npy',pool1_error)
#
#
conv2_1 = feature6.f6(pool1)
conv2_1_1 = feature6_1.f6_1(pool1_1)
conv2_1_2 = feature6_2.f6_2(pool1_2)
conv2_1_dim=np.shape(conv2_1_2)    #same_dimension
conv2_1_b= np.load('conv2_1_b.npy')   #load_bias
conv2_1_b=conv2_1_b.reshape(conv2_1_dim[0],conv2_1_dim[1],1,1)   #two_dim to four_dim
conv2_1_2=conv2_1_2-conv2_1_b    #sub_bias
conv2_1_error=conv2_1-(conv2_1_1+conv2_1_2)
np.save('conv2_1.npy',conv2_1)
np.save('conv2_1_1.npy',conv2_1_1)
np.save('conv2_1_2.npy',conv2_1_2)
np.save('conv2_1_error.npy',conv2_1_error)

relu2_1 = feature7.f7(conv2_1)
relu2_1_1 = feature7_1.f7_1(conv2_1_1, conv2_1_2)
relu2_1_2 = feature7_2.f7_2(conv2_1_1, conv2_1_2)
relu2_1_error=relu2_1-(relu2_1_1+relu2_1_2)
np.save('relu2_1.npy',relu2_1)
np.save('relu2_1_1.npy',relu2_1_1)
np.save('relu2_1_2.npy',relu2_1_2)
np.save('relu2_1_error.npy',relu2_1_error)

conv2_2 = feature8.f8(relu2_1)
conv2_2_1 = feature8_1.f8_1(relu2_1_1)
conv2_2_2 = feature8_2.f8_2(relu2_1_2)
conv2_2_dim=np.shape(conv2_2_2)    #same_dimension
conv2_2_b= np.load('conv2_2_b.npy')   #load_bias
conv2_2_b=conv2_2_b.reshape(conv2_2_dim[0],conv2_2_dim[1],1,1)   #two_dim to four_dim
conv2_2_2=conv2_2_2-conv2_2_b    #sub_bias
conv2_2_error=conv2_2-(conv2_2_1+conv2_2_2)
np.save('conv2_2.npy',conv2_2)
np.save('conv2_2_1.npy',conv2_2_1)
np.save('conv2_2_2.npy',conv2_2_2)
np.save('conv2_2_error.npy',conv2_2_error)

relu2_2 = feature9.f9(conv2_2)
relu2_2_1 = feature9_1.f9_1(conv2_2_1, conv2_2_2)
relu2_2_2 = feature9_2.f9_2(conv2_2_1, conv2_2_2)
relu2_2_error=relu2_2-(relu2_2_1+relu2_2_2)
np.save('relu2_2.npy',relu2_2)
np.save('relu2_2_1.npy',relu2_2_1)
np.save('relu2_2_2.npy',relu2_2_2)
np.save('relu2_2_error.npy',relu2_2_error)

pool2 = feature10.f10(relu2_2)
pool2_1 = feature10_1.f10_1(relu2_2_1,relu2_2_2)
pool2_2 = feature10_2.f10_2(relu2_2_1,relu2_2_2)
pool2_error=pool2-(pool2_1+pool2_2)
np.save('pool2.npy',pool2)
np.save('pool2_1.npy',pool2_1)
np.save('pool2_2.npy',pool2_2)
np.save('pool2_error.npy',pool2_error)

conv3_1 = feature11.f11(pool2)
conv3_1_1 = feature11_1.f11_1(pool2_1)
conv3_1_2 = feature11_2.f11_2(pool2_2)
conv3_1_dim=np.shape(conv3_1_2)    #same_dimension
conv3_1_b= np.load('conv3_1_b.npy')   #load_bias
conv3_1_b=conv3_1_b.reshape(conv3_1_dim[0],conv3_1_dim[1],1,1)   #two_dim to four_dim
conv3_1_2=conv3_1_2-conv3_1_b    #sub_bias
conv3_1_error=conv3_1-(conv3_1_1+conv3_1_2)
np.save('conv3_1.npy',conv3_1)
np.save('conv3_1_1.npy',conv3_1_1)
np.save('conv3_1_2.npy',conv3_1_2)
np.save('conv3_1_error.npy',conv3_1_error)

relu3_1 = feature12.f12(conv3_1)
relu3_1_1 = feature12_1.f12_1(conv3_1_1, conv3_1_2)
relu3_1_2 = feature12_2.f12_2(conv3_1_1, conv3_1_2)
relu3_1_error=relu3_1-(relu3_1_1+relu3_1_2)
np.save('relu3_1.npy',relu3_1)
np.save('relu3_1_1.npy',relu3_1_1)
np.save('relu3_1_2.npy',relu3_1_2)
np.save('relu3_1_error.npy',relu3_1_error)

conv3_2 = feature13.f13(relu3_1)
conv3_2_1 = feature13_1.f13_1(relu3_1_1)
conv3_2_2 = feature13_2.f13_2(relu3_1_2)
conv3_2_dim=np.shape(conv3_2_2)    #same_dimension
conv3_2_b= np.load('conv3_2_b.npy')   #load_bias
conv3_2_b=conv3_2_b.reshape(conv3_2_dim[0],conv3_2_dim[1],1,1)   #two_dim to four_dim
conv3_2_2=conv3_2_2-conv3_2_b    #sub_bias
conv3_2_error=conv3_2-(conv3_2_1+conv3_2_2)
np.save('conv3_2.npy',conv3_2)
np.save('conv3_2_1.npy',conv3_2_1)
np.save('conv3_2_2.npy',conv3_2_2)
np.save('conv3_2_error.npy',conv3_2_error)

relu3_2 = feature14.f14(conv3_2)
relu3_2_1 = feature14_1.f14_1(conv3_2_1, conv3_2_2)
relu3_2_2 = feature14_2.f14_2(conv3_2_1, conv3_2_2)
relu3_2_error=relu3_2-(relu3_2_1+relu3_2_2)
np.save('relu3_2.npy',relu3_2)
np.save('relu3_2_1.npy',relu3_2_1)
np.save('relu3_2_2.npy',relu3_2_2)
np.save('relu3_2_error.npy',relu3_2_error)

conv3_3 = feature15.f15(relu3_2)
conv3_3_1 = feature15_1.f15_1(relu3_2_1)
conv3_3_2 = feature15_2.f15_2(relu3_2_2)
conv3_3_dim=np.shape(conv3_3_2)    #same_dimension
conv3_3_b= np.load('conv3_3_b.npy')   #load_bias
conv3_3_b=conv3_3_b.reshape(conv3_3_dim[0],conv3_3_dim[1],1,1)   #two_dim to four_dim
conv3_3_2=conv3_3_2-conv3_3_b    #sub_bias
conv3_3_error=conv3_3-(conv3_3_1+conv3_3_2)
np.save('conv3_3.npy',conv3_3)
np.save('conv3_3_1.npy',conv3_3_1)
np.save('conv3_3_2.npy',conv3_3_2)
np.save('conv3_3_error.npy',conv3_3_error)

relu3_3 = feature16.f16(conv3_3)
relu3_3_1 = feature16_1.f16_1(conv3_3_1, conv3_3_2)
relu3_3_2 = feature16_2.f16_2(conv3_3_1, conv3_3_2)
relu3_3_error=relu3_3-(relu3_3_1+relu3_3_2)
np.save('relu3_3.npy',relu3_3)
np.save('relu3_3_1.npy',relu3_3_1)
np.save('relu3_3_2.npy',relu3_3_2)
np.save('relu3_3_error.npy',relu3_3_error)

pool3 = feature17.f17(relu3_3)
pool3_1 = feature17_1.f17_1(relu3_3_1, relu3_3_2)
pool3_2 = feature17_2.f17_2(relu3_3_1, relu3_3_2)
pool3_error=pool3 -(pool3_1+pool3_2)
np.save('pool3.npy',pool3)
np.save('pool3_1.npy',pool3_1)
np.save('pool3_2.npy',pool3_2)
np.save('pool3_error.npy',pool3_error)

conv4_1 = feature18.f18(pool3)
conv4_1_1 = feature18_1.f18_1(pool3_1)
conv4_1_2 = feature18_2.f18_2(pool3_2)
conv4_1_dim=np.shape(conv4_1_2)    #same_dimension
conv4_1_b= np.load('conv4_1_b.npy')   #load_bias
conv4_1_b=conv4_1_b.reshape(conv4_1_dim[0],conv4_1_dim[1],1,1)   #two_dim to four_dim
conv4_1_2=conv4_1_2-conv4_1_b    #sub_bias
conv4_1_error=conv4_1-(conv4_1_1+conv4_1_2)
np.save('conv4_1.npy',conv4_1)
np.save('conv4_1_1.npy',conv4_1_1)
np.save('conv4_1_2.npy',conv4_1_2)
np.save('conv4_1_error.npy',conv4_1_error)

relu4_1 = feature19.f19(conv4_1)
relu4_1_1 = feature19_1.f19_1(conv4_1_1, conv4_1_2)
relu4_1_2 = feature19_2.f19_2(conv4_1_1, conv4_1_2)
relu4_1_error=relu4_1-(relu4_1_1+relu4_1_2)
np.save('relu4_1.npy',relu4_1)
np.save('relu4_1_1.npy',relu4_1_1)
np.save('relu4_1_2.npy',relu4_1_2)
np.save('relu4_1_error.npy',relu4_1_error)

conv4_2 = feature20.f20(relu4_1)
conv4_2_1 = feature20_1.f20_1(relu4_1_1)
conv4_2_2 = feature20_2.f20_2(relu4_1_2)
conv4_2_dim=np.shape(conv4_2_2)    #same_dimension
conv4_2_b= np.load('conv4_2_b.npy')   #load_bias
conv4_2_b=conv4_2_b.reshape(conv4_2_dim[0],conv4_2_dim[1],1,1)   #two_dim to four_dim
conv4_2_2=conv4_2_2-conv4_2_b    #sub_bias
conv4_2_error=conv4_2-(conv4_2_1+conv4_2_2)
np.save('conv4_2.npy',conv4_2)
np.save('conv4_2_1.npy',conv4_2_1)
np.save('conv4_2_2.npy',conv4_2_2)
np.save('conv4_2_error.npy',conv4_2_error)

relu4_2 = feature21.f21(conv4_2)
relu4_2_1 = feature21_1.f21_1(conv4_2_1, conv4_2_2)
relu4_2_2 = feature21_2.f21_2(conv4_2_1, conv4_2_2)
relu4_2_error=relu4_2-(relu4_2_1+relu4_2_2)
np.save('relu4_2.npy',relu4_2)
np.save('relu4_2_1.npy',relu4_2_1)
np.save('relu4_2_2.npy',relu4_2_2)
np.save('relu4_2_error.npy',relu4_2_error)

conv4_3 = feature22.f22(relu4_2)
conv4_3_1 = feature22_1.f22_1(relu4_2_1)
conv4_3_2 = feature22_2.f22_2(relu4_2_2)
conv4_3_dim=np.shape(conv4_3_2)    #same_dimension
conv4_3_b= np.load('conv4_3_b.npy')   #load_bias
conv4_3_b=conv4_3_b.reshape(conv4_3_dim[0],conv4_3_dim[1],1,1)   #two_dim to four_dim
conv4_3_2=conv4_3_2-conv4_3_b    #sub_bias
conv4_3_error=conv4_3-(conv4_3_1+conv4_3_2)
np.save('conv4_3.npy',conv4_3)
np.save('conv4_3_1.npy',conv4_3_1)
np.save('conv4_3_2.npy',conv4_3_2)
np.save('conv4_3_error.npy',conv4_3_error)

relu4_3 = feature23.f23(conv4_3)
relu4_3_1 = feature23_1.f23_1(conv4_3_1, conv4_3_2)
relu4_3_2 = feature23_2.f23_2(conv4_3_1, conv4_3_2)
relu4_3_error=relu4_3-(relu4_3_1+relu4_3_2)
np.save('relu4_3.npy',relu4_3)
np.save('relu4_3_1.npy',relu4_3_1)
np.save('relu4_3_2.npy',relu4_3_2)
np.save('relu4_3_error.npy',relu4_3_error)

pool4 = feature24.f24(relu4_3)
pool4_1 = feature24_1.f24_1(relu4_3_1, relu4_3_2)
pool4_2 = feature24_2.f24_2(relu4_3_1, relu4_3_2)
pool4_error=pool4 -(pool4_1+pool4_2)
np.save('pool4.npy',pool4)
np.save('pool4_1.npy',pool4_1)
np.save('pool4_2.npy',pool4_2)
np.save('pool4_error.npy',pool4_error)

conv5_1 = feature25.f25(pool4)
conv5_1_1 = feature25_1.f25_1(pool4_1)
conv5_1_2 = feature25_2.f25_2(pool4_2)
conv5_1_dim=np.shape(conv5_1_2)    #same_dimension
conv5_1_b= np.load('conv5_1_b.npy')   #load_bias
conv5_1_b=conv5_1_b.reshape(conv5_1_dim[0],conv5_1_dim[1],1,1)   #two_dim to four_dim
conv5_1_2=conv5_1_2-conv5_1_b    #sub_bias
conv5_1_error=conv5_1-(conv5_1_1+conv5_1_2)
np.save('conv5_1.npy',conv5_1)
np.save('conv5_1_1.npy',conv5_1_1)
np.save('conv5_1_2.npy',conv5_1_2)
np.save('conv5_1_error.npy',conv5_1_error)

relu5_1 = feature26.f26(conv5_1)
relu5_1_1 = feature26_1.f26_1(conv5_1_1, conv5_1_2)
relu5_1_2 = feature26_2.f26_2(conv5_1_1, conv5_1_2)
relu5_1_error=relu5_1-(relu5_1_1+relu5_1_2)
np.save('relu5_1.npy',relu5_1)
np.save('relu5_1_1.npy',relu5_1_1)
np.save('relu5_1_2.npy',relu5_1_2)
np.save('relu5_1_error.npy',relu5_1_error)

conv5_2 = feature27.f27(relu5_1)
conv5_2_1 = feature27_1.f27_1(relu5_1_1)
conv5_2_2 = feature27_2.f27_2(relu5_1_2)
conv5_2_dim=np.shape(conv5_2_2)    #same_dimension
conv5_2_b= np.load('conv5_2_b.npy')   #load_bias
conv5_2_b=conv5_2_b.reshape(conv5_2_dim[0],conv5_2_dim[1],1,1)   #two_dim to four_dim
conv5_2_2=conv5_2_2-conv5_2_b    #sub_bias
conv5_2_error=conv5_2-(conv5_2_1+conv5_2_2)
np.save('conv5_2.npy',conv5_2)
np.save('conv5_2_1.npy',conv5_2_1)
np.save('conv5_2_2.npy',conv5_2_2)
np.save('conv5_2_error.npy',conv5_2_error)

relu5_2 = feature28.f28(conv5_2)
relu5_2_1 = feature28_1.f28_1(conv5_2_1, conv5_2_2)
relu5_2_2 = feature28_2.f28_2(conv5_2_1, conv5_2_2)
relu5_2_error=relu5_2-(relu5_2_1+relu5_2_2)
np.save('relu5_2.npy',relu5_2)
np.save('relu5_2_1.npy',relu5_2_1)
np.save('relu5_2_2.npy',relu5_2_2)
np.save('relu5_2_error.npy',relu5_2_error)

conv5_3 = feature29.f29(relu5_2)
conv5_3_1 = feature29_1.f29_1(relu5_2_1)
conv5_3_2 = feature29_2.f29_2(relu5_2_2)
conv5_3_dim=np.shape(conv5_3_2)    #same_dimension
conv5_3_b= np.load('conv5_3_b.npy')   #load_bias
conv5_3_b=conv5_3_b.reshape(conv5_3_dim[0],conv5_3_dim[1],1,1)   #two_dim to four_dim
conv5_3_2=conv5_3_2-conv5_3_b    #sub_bias
conv5_3_error=conv5_3-(conv5_3_1+conv5_3_2)
np.save('conv5_3.npy',conv5_3)
np.save('conv5_3_1.npy',conv5_3_1)
np.save('conv5_3_2.npy',conv5_3_2)
np.save('conv5_3_error.npy',conv5_3_error)

relu5_3 = feature30.f30(conv5_3)
relu5_3_1 = feature30_1.f30_1(conv5_3_1, conv5_3_2)
relu5_3_2 = feature30_2.f30_2(conv5_3_1, conv5_3_2)
relu5_3_error=relu5_3-(relu5_3_1+relu5_3_2)
np.save('relu5_3.npy',relu5_3)
np.save('relu5_3_1.npy',relu5_3_1)
np.save('relu5_3_2.npy',relu5_3_2)
np.save('relu5_3_error.npy',relu5_3_error)

pool5 = feature31.f31(relu5_3)
pool5_1 = feature31_1.f31_1(relu5_3_1, relu5_3_2)
pool5_2 = feature31_2.f31_2(relu5_3_1, relu5_3_2)
pool5_error=pool5 -(pool5_1+pool5_2)
np.save('pool5.npy',pool5)
np.save('pool5_1.npy',pool5_1)
np.save('pool5_2.npy',pool5_2)
np.save('pool5_error.npy',pool5_error)

fc6 = feature32.f32(pool5)
fc6_1 = feature32_1.f32_1(pool5_1)
fc6_2 = feature32_2.f32_2(pool5_2)
fc6_b= np.load('fc6_b.npy')   #load_bias
fc6_2=fc6_2-fc6_b    #sub_bias
fc6_error=fc6-(fc6_1+fc6_2)
np.save('fc6.npy',fc6)
np.save('fc6_1.npy',fc6_1)
np.save('fc6_2npy',fc6_2)
np.save('fc6_error.npy',fc6_error)

relu6 = feature33.f33(fc6)
relu6_1 = feature33_1.f33_1(fc6_1, fc6_2)
relu6_2 = feature33_2.f33_2(fc6_1, fc6_2)
relu6_error=relu6-(relu6_1+relu6_2)
np.save('relu6.npy',relu6)
np.save('relu6_1.npy',relu6_1)
np.save('relu6_2npy',relu6_2)
np.save('relu6_error.npy',relu6_error)

fc7 = feature34.f34(relu6)
fc7_1 = feature34_1.f34_1(relu6_1)
fc7_2 = feature34_2.f34_2(relu6_2)
fc7_b= np.load('fc7_b.npy')   #load_bias
fc7_2=fc7_2-fc7_b    #sub_bias
fc7_error=fc7-(fc7_1+fc7_2)
np.save('fc7.npy',fc7)
np.save('fc7_1.npy',fc7_1)
np.save('fc7_2npy',fc7_2)
np.save('fc7_error.npy',fc7_error)

relu7 = feature35.f35(fc7)
relu7_1 = feature35_1.f35_1(fc7_1, fc7_2)
relu7_2 = feature35_2.f35_2(fc7_1, fc7_2)
relu7_error=relu7-(relu7_1+relu7_2)
np.save('relu7.npy',relu7)
np.save('relu7_1.npy',relu7_1)
np.save('relu7_2npy',relu7_2)
np.save('relu7_error.npy',relu7_error)

score = feature36.f36(relu7)
score1 = feature36_1.f36_1(relu7_1)
score2 = feature36_2.f36_2(relu7_2)
score_b= np.load('score_b.npy')   #load_bias
score2=score2-score_b    #sub_bias
score_inter = score1 + score2
score_error=score - score_inter
np.save('score.npy',score)
np.save('score1.npy',score1)
np.save('score2npy',score2)
np.save('score_inter.npy',score_inter)
np.save('score_error.npy',score_error)

ori_class=score.argmax()
div_class=score_inter.argmax()
np.save('ori_class.npy',ori_class)
np.save('div_class.npy',div_class)

print('..................class_begin......................')
print(ori_class)
print('.............................divided_before_after............................................')
print(div_class)
print('..................class_end..................')

t2 = time.time()
print('..................time_bigin..................')
print(t2-t1)
print('..................time_end..................')

#rgb_data/testing/3.png