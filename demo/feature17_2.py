import numpy as np
import pylab
import pickle as p
import smc_function as smc
import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()

def f17_2(relu3_3_1, relu3_3_2):
	model_def = 'deploy17_2.prototxt'
	#model_weights = 'test.caffemodel'

	net = caffe.Net(model_def,
		            #model_weights,
		            caffe.TEST)

	net.blobs['relu3_3_1'].data[...] = relu3_3_1
	net.blobs['relu3_3_2'].data[...] = relu3_3_2

	net.forward()

	output = net.blobs['pool3_2'].data

	return output
