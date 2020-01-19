import numpy as np
import pylab
import pickle as p

import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()

def f16_2(conv3_3_1, conv3_3_2):
	model_def = 'deploy16_2.prototxt'
	#model_weights = 'test.caffemodel'

	net = caffe.Net(model_def,
		            #model_weights,
		            caffe.TEST)
	#構造相同的維度
	net.blobs['conv3_3_1'].data[...] = conv3_3_1
	net.blobs['conv3_3_2'].data[...] = conv3_3_2
	net.forward()

	output = net.blobs['relu3_3_2'].data
	
	return output
