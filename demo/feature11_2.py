import numpy as np
import pylab
import pickle as p
import time

import sys

caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
import caffe

caffe.set_mode_gpu()

def f11_2(pool2_2):
	model_def = 'deploy11_2.prototxt'
	model_weights = 'revised11.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['pool2_2'].data[...] = pool2_2

	net.forward()

	output = net.blobs['conv3_1_2'].data

	return output
