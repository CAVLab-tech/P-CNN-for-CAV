import numpy as np
import pylab
import pickle as p

import sys
# caffe_root = '../'
# sys.path.insert(0, caffe_root + 'python')

caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
import caffe

caffe.set_mode_gpu()


def f2_1(conv1_1_1, conv1_1_2):
	model_def = 'deploy2_1.prototxt'
	#model_weights = 'test.caffemodel'

	net = caffe.Net(model_def,
					#model_weights,
					caffe.TEST)

	net.blobs['conv1_1_1'].data[...] = conv1_1_1
	net.blobs['conv1_1_2'].data[...] = conv1_1_2
	net.forward()

	output = net.blobs['relu1_1_1'].data
	return output
