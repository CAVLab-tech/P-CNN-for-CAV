import numpy as np
import pylab
import pickle as p
import time

import sys

caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
import caffe

caffe.set_mode_gpu()

def f11_1(pool2_1):
	model_def = 'deploy11_1.prototxt'
	model_weights = 'revised11.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['pool2_1'].data[...] = pool2_1

	net.forward()

	output = net.blobs['conv3_1_1'].data

	return output
