import numpy as np
import pylab
import pickle as p

import sys
caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
import caffe

caffe.set_mode_gpu()

def f7_1(conv2_1_1,conv2_1_2):
	model_def = 'deploy7_1.prototxt'
	#model_weights = 'test.caffemodel'

	net = caffe.Net(model_def,
		            #model_weights,
		            caffe.TEST)
	#構造相同的維度
	net.blobs['conv2_1_1'].data[...] = conv2_1_1
	net.blobs['conv2_1_2'].data[...] = conv2_1_2
	net.forward()

	output = net.blobs['relu2_1_1'].data
	
	return output
