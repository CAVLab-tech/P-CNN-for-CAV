import numpy as np
import pylab
import pickle as p
import cv2
import time

import sys

# caffe_root = '/home/cqx/caffe/'
# sys.path.append('/home/cqx/caffe/python')

caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')

import caffe
caffe.set_mode_gpu()



image_in = cv2.imread('rgb_data/testing/4.png')  #读取图片像素值
crop_size = (224, 224)
image = cv2.resize(image_in, crop_size, interpolation = cv2.INTER_AREA)     #转换成224×224×3的维度

image=image.reshape(224,224,-1)
image = image.transpose(2, 0, 1)



def f1(image):
	model_def = 'deploy1.prototxt'
	model_weights = 'revised1.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['data'].data[...] = image

	net.forward()
	output = net.blobs['conv1_1'].data
	return output

	return net.blobs['data'].data[...]


t1 = time.time()


f1(image)


t2 = time.time()
print(t2-t1)