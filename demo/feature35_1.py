import numpy as np
import pylab
import pickle as p

import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()

def f35_1(fc7_1, fc7_2):
	model_def = 'deploy35_1.prototxt'
	#model_weights = 'test.caffemodel'

	net = caffe.Net(model_def,
		            #model_weights,
		            caffe.TEST)
	#構造相同的維度
	fc7_1 = fc7_1.reshape(1, 4096, 1, 1)
	fc7_2 = fc7_2.reshape(1, 4096, 1, 1)
	net.blobs['fc7_1'].data[...] = fc7_1
	net.blobs['fc7_2'].data[...] = fc7_2

	net.forward()
	output = net.blobs['relu7_1'].data
	
	return output