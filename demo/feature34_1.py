import numpy as np
import pylab
import pickle as p

import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()

def f34_1(relu6_1):
	model_def = 'deploy34_1.prototxt'
	model_weights = 'revised34.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	relu6_1= relu6_1.reshape(1, 4096, 1,1)
	net.blobs['relu6_1'].data[...] = relu6_1

	net.forward()

	output = net.blobs['fc7_1'].data

	return output