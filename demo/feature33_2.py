import numpy as np
import pylab
import pickle as p

import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()

def f33_2(fc6_1,fc6_2):
	model_def = 'deploy33_2.prototxt'
	#model_weights = 'test.caffemodel'

	net = caffe.Net(model_def,
		            #model_weights,
		            caffe.TEST)
	#構造相同的維度
	fc6_1 = fc6_1.reshape(1, 4096, 1,1)
	fc6_2 = fc6_2.reshape(1, 4096, 1,1)
	net.blobs['fc6_1'].data[...] = fc6_1
	net.blobs['fc6_2'].data[...] = fc6_2

	net.forward()
	output = net.blobs['relu6_2'].data
	
	return output