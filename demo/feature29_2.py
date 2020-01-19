import numpy as np
import pylab
import pickle as p
import time

import sys

caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
import caffe

caffe.set_mode_gpu()

def f29_2(relu5_2_2):
	model_def = 'deploy29_2.prototxt'
	model_weights = 'revised29.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['relu5_2_2'].data[...] = relu5_2_2

	net.forward()

	output = net.blobs['conv5_3_2'].data

	return output
