import numpy as np
import pylab
import pickle as p
import time

import sys

caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
import caffe

caffe.set_mode_gpu()

def f27_1(relu5_1_1):
	model_def = 'deploy27_1.prototxt'
	model_weights = 'revised27.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['relu5_1_1'].data[...] = relu5_1_1

	net.forward()

	output = net.blobs['conv5_2_1'].data

	return output
