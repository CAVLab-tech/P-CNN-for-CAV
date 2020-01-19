import numpy as np
import pylab
import pickle as p
import time

import sys

caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
import caffe

caffe.set_mode_gpu()

def f13_2(relu3_1_2):
	model_def = 'deploy13_2.prototxt'
	model_weights = 'revised13.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['relu3_1_2'].data[...] = relu3_1_2

	net.forward()

	output = net.blobs['conv3_2_2'].data

	return output
