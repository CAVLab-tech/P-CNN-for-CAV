import numpy as np
import pylab
import pickle as p

import sys
caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
import caffe

caffe.set_mode_gpu()

def f7(conv2_1):
	model_def = 'deploy7.prototxt'
	#model_weights = 'test.caffemodel'

	net = caffe.Net(model_def,
		           # model_weights,
		            caffe.TEST)
	#構造相同的維度
	net.blobs['conv2_1'].data[...] = conv2_1

	net.forward()

	output = net.blobs['relu2_1'].data
	
	return output