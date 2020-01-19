import numpy as np
import pylab
import pickle as p
import time

import sys

caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
import caffe

caffe.set_mode_gpu()

def f3_1(relu1_1_1):
	model_def = 'deploy3_1.prototxt'
	model_weights = 'revised3.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['relu1_1_1'].data[...] = relu1_1_1

	net.forward()

	output = net.blobs['conv1_2_1'].data

	return output
