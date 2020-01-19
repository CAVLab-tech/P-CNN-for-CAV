import numpy as np
import pylab
import pickle as p

import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()

def f32(pool5):
	model_def = 'deploy32.prototxt'
	model_weights = 'revised32.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['pool5'].data[...] = pool5

	net.forward()

	output = net.blobs['fc6'].data

	return output