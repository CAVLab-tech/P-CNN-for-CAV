import numpy as np
import pylab
import pickle as p
import time

import sys

# caffe_root = '/home/cqx/caffe/'
# sys.path.append('/home/cqx/caffe/python')

caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')

import caffe

caffe.set_mode_gpu()



def f1_1(image1):
	model_def = 'deploy1_1.prototxt'
	model_weights = 'revised1.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['data1'].data[...] = image1

	net.forward()

	output = net.blobs['conv1_1_1'].data

	return output
