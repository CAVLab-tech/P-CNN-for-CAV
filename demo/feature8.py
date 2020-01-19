import numpy as np
import pylab
import pickle as p
import time

import sys

caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
import caffe

caffe.set_mode_gpu()

def f8(relu2_1):
	model_def = 'deploy8.prototxt'
	model_weights = 'revised8.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['relu2_1'].data[...] = relu2_1

	net.forward()

	output = net.blobs['conv2_2'].data

	return output