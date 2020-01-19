import numpy as np
import pylab
import pickle as p
import time

import sys

caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
import caffe

caffe.set_mode_gpu()

def f3_2(relu1_1_2):
	model_def = 'deploy3_2.prototxt'
	model_weights = 'revised3.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['relu1_1_2'].data[...] = relu1_1_2

	net.forward()

	output = net.blobs['conv1_2_2'].data

	return output
