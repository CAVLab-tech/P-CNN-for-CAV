import numpy as np
import pylab
import pickle as p
import time

import sys

caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
import caffe

caffe.set_mode_gpu()

def f22(relu4_2):
	model_def = 'deploy22.prototxt'
	model_weights = 'revised22.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['relu4_2'].data[...] = relu4_2

	net.forward()

	output = net.blobs['conv4_3'].data

	return output