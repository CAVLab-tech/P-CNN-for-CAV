import numpy as np
import pylab
import pickle as p

import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()

def f33(fc6):
	model_def = 'deploy33.prototxt'
	#model_weights = 'test.caffemodel'

	net = caffe.Net(model_def,
		            #model_weights,
		            caffe.TEST)
	#構造相同的維度
	fc6 = fc6.reshape(1, 4096, 1,1)
	net.blobs['fc6'].data[...] = fc6


	net.forward()

	output = net.blobs['relu6'].data
	
	return output