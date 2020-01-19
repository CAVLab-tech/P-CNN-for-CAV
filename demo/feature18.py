import numpy as np
import pylab
import pickle as p
import time

import sys

caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
import caffe

caffe.set_mode_gpu()

def f18(pool3):
	model_def = 'deploy18.prototxt'
	model_weights = 'revised18.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['pool3'].data[...] = pool3

	net.forward()

	output = net.blobs['conv4_1'].data

	return output