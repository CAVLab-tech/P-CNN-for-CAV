import numpy as np
import pylab
import pickle as p

import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()

def f32_1(pool5_1):
	model_def = 'deploy32_1.prototxt'
	model_weights = 'revised32.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['pool5_1'].data[...] = pool5_1

	net.forward()

	output = net.blobs['fc6_1'].data

	return output