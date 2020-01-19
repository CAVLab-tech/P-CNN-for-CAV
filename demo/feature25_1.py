import numpy as np
import pylab
import pickle as p
import time

import sys

caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
import caffe

caffe.set_mode_gpu()

def f25_1(pool4_1):
	model_def = 'deploy25_1.prototxt'
	model_weights = 'revised25.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['pool4_1'].data[...] = pool4_1

	net.forward()

	output = net.blobs['conv5_1_1'].data

	return output
