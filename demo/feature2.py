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


def f2(conv1_1):
	model_def = 'deploy2.prototxt'
	#model_weights = 'test.caffemodel'

	net = caffe.Net(model_def,
		            #model_weights,
		            caffe.TEST)
	#構造相同的維度
	net.blobs['conv1_1'].data[...] = conv1_1

	net.forward()

	output = net.blobs['relu1_1'].data
	
	return output