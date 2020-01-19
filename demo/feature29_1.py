import numpy as np
import pylab
import pickle as p
import time

import sys

caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
import caffe

caffe.set_mode_gpu()

def f29_1(relu5_2_1):
	model_def = 'deploy29_1.prototxt'
	model_weights = 'revised29.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['relu5_2_1'].data[...] = relu5_2_1

	net.forward()

	output = net.blobs['conv5_3_1'].data

	return output
