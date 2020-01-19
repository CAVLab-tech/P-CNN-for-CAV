import numpy as np
import pylab
import pickle as p
import time

import sys

caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
import caffe

caffe.set_mode_gpu()

def f25(pool4):
	model_def = 'deploy25.prototxt'
	model_weights = 'revised25.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['pool4'].data[...] = pool4

	net.forward()

	output = net.blobs['conv5_1'].data

	return output