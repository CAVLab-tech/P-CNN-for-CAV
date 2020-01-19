import numpy as np
import pylab
import pickle as p
import time

import sys

caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
import caffe

caffe.set_mode_gpu()

def f15(relu3_2):
	model_def = 'deploy15.prototxt'
	model_weights = 'revised15.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['relu3_2'].data[...] = relu3_2

	net.forward()

	output = net.blobs['conv3_3'].data

	return output