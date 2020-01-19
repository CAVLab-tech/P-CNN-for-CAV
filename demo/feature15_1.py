import numpy as np
import pylab
import pickle as p
import time

import sys

caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
import caffe

caffe.set_mode_gpu()

def f15_1(relu3_2_1):
	model_def = 'deploy15_1.prototxt'
	model_weights = 'revised15.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['relu3_2_1'].data[...] = relu3_2_1

	net.forward()

	output = net.blobs['conv3_3_1'].data

	return output
