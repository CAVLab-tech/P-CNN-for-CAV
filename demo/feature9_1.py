import numpy as np
import pylab
import pickle as p

import sys
caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
import caffe

caffe.set_mode_gpu()

def f9_1(conv2_2_1,conv2_2_2):
	model_def = 'deploy9_1.prototxt'
	#model_weights = 'test.caffemodel'

	net = caffe.Net(model_def,
		            #model_weights,
		            caffe.TEST)
	#構造相同的維度
	net.blobs['conv2_2_1'].data[...] = conv2_2_1
	net.blobs['conv2_2_2'].data[...] = conv2_2_2

	net.forward()

	output = net.blobs['relu2_2_1'].data
	
	return output
