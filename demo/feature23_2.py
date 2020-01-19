import numpy as np
import pylab
import pickle as p

import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()

def f23_2(conv4_3_1,conv4_3_2):
	model_def = 'deploy23_2.prototxt'
	#model_weights = 'test.caffemodel'

	net = caffe.Net(model_def,
		            #model_weights,
		            caffe.TEST)
	#構造相同的維度
	net.blobs['conv4_3_1'].data[...] = conv4_3_1
	net.blobs['conv4_3_2'].data[...] = conv4_3_2

	net.forward()

	output = net.blobs['relu4_3_2'].data
	
	return output
