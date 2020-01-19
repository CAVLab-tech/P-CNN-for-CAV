import numpy as np
import pylab
import pickle as p
import time

import sys

caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
import caffe

caffe.set_mode_gpu()

def f6_2(pool1_2):
	model_def = 'deploy6_2.prototxt'
	model_weights = 'revised6.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['pool1_2'].data[...] = pool1_2

	net.forward()

	output = net.blobs['conv2_1_2'].data

	return output
