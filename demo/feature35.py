import numpy as np
import pylab
import pickle as p

import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()

def f35(fc7):
	model_def = 'deploy35.prototxt'
	#model_weights = 'test.caffemodel'

	net = caffe.Net(model_def,
		            #model_weights,
		            caffe.TEST)
	#構造相同的維度
	fc7 = fc7.reshape(1, 4096, 1, 1)
	net.blobs['fc7'].data[...] = fc7

	net.forward()

	output = net.blobs['relu7'].data
	
	return output