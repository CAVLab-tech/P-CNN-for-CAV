import numpy as np
import pylab
import pickle as p

import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()

def f9_2(conv2_2_1,conv2_2_2):
	model_def = 'deploy9_2.prototxt'
	#model_weights = 'test.caffemodel'

	net = caffe.Net(model_def,
		            #model_weights,
		            caffe.TEST)
	#構造相同的維度
	net.blobs['conv2_2_1'].data[...] = conv2_2_1
	net.blobs['conv2_2_2'].data[...] = conv2_2_2

	net.forward()

	output = net.blobs['relu2_2_2'].data
	
	return output
