import numpy as np
import pylab
import pickle as p
import time

import sys

caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
import caffe

caffe.set_mode_gpu()

def f27_2(relu5_1_2):
	model_def = 'deploy27_2.prototxt'
	model_weights = 'revised27.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['relu5_1_2'].data[...] = relu5_1_2

	net.forward()

	output = net.blobs['conv5_2_2'].data

	return output
