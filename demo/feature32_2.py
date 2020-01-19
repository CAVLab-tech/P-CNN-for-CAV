import numpy as np
import pylab
import pickle as p

import sys
caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
import caffe

caffe.set_mode_gpu()

def f32_2(pool5_2):
	model_def = 'deploy32_2.prototxt'
	model_weights = 'revised32.caffemodel'

	net = caffe.Net(model_def,
					model_weights,
		            caffe.TEST)

	net.blobs['pool5_2'].data[...] = pool5_2

	net.forward()

	output = net.blobs['fc6_2'].data

	return output