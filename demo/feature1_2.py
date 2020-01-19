import numpy as np
import pylab
import pickle as p
import time

import sys

# caffe_root = '/home/cqx/caffe/'
# sys.path.append('/home/cqx/caffe/python')

caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')

import caffe

caffe.set_mode_gpu()



def f1_2(image2):
	model_def = 'deploy1_2.prototxt'
	model_weights = 'revised1.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['data2'].data[...] = image2

	net.forward()

	output = net.blobs['conv1_1_2'].data

	return output
