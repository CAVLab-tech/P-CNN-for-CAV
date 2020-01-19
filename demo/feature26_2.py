import numpy as np
import pylab
import pickle as p

import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()

def f26_2(conv5_1_1,conv5_1_2):
	model_def = 'deploy26_2.prototxt'
	#model_weights = 'test.caffemodel'

	net = caffe.Net(model_def,
		            #model_weights,
		            caffe.TEST)
	#構造相同的維度
	net.blobs['conv5_1_1'].data[...] = conv5_1_1
	net.blobs['conv5_1_2'].data[...] = conv5_1_2

	net.forward()

	output = net.blobs['relu5_1_2'].data
	
	return output
