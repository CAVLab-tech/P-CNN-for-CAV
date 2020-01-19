import numpy as np
import pylab
import pickle as p

import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()

def f36(relu7):
	model_def = 'deploy36.prototxt'
	model_weights = 'revised36.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)


    #relu7 = relu7.reshape(1, 4096, 1, 1)
	net.blobs['relu7'].data[...] = relu7

	net.forward()

	output = net.blobs['score'].data

	return output