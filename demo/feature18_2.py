import numpy as np
import pylab
import pickle as p
import time

import sys

caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
import caffe

caffe.set_mode_gpu()

def f18_2(pool3_2):
	model_def = 'deploy18_2.prototxt'
	model_weights = 'revised18.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['pool3_2'].data[...] = pool3_2

	net.forward()

	output = net.blobs['conv4_1_2'].data

	return output
