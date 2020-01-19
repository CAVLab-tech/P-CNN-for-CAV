import numpy as np
import pylab
import pickle as p

import sys
caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
import caffe

caffe.set_mode_gpu()

def f4_2(conv1_2_1,conv1_2_2):
	model_def = 'deploy4_2.prototxt'
	#model_weights = 'test.caffemodel'

	net = caffe.Net(model_def,
		            #model_weights,
		            caffe.TEST)
	#構造相同的維度
	net.blobs['conv1_2_1'].data[...] = conv1_2_1
	net.blobs['conv1_2_2'].data[...] = conv1_2_2

	net.forward()

	output = net.blobs['relu1_2_2'].data
	
	return output
