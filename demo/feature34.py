import numpy as np
import pylab
import pickle as p

import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()

def f34(relu6):
	model_def = 'deploy34.prototxt'
	model_weights = 'revised34.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	relu6= relu6.reshape(1, 4096, 1,1)
	net.blobs['relu6'].data[...] = relu6

	net.forward()

	output = net.blobs['fc7'].data

	return output