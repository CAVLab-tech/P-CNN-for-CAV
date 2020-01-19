import numpy as np
import pylab
import pickle as p

import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()

def f34_2(relu6_2):
	model_def = 'deploy34_2.prototxt'
	model_weights = 'revised34.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	relu6_2= relu6_2.reshape(1, 4096, 1,1)
	net.blobs['relu6_2'].data[...] = relu6_2

	net.forward()

	output = net.blobs['fc7_2'].data

	return output