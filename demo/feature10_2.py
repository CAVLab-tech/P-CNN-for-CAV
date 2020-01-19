import numpy as np
import pylab
import pickle as p
import smc_function as smc
import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()

def f10_2(relu2_2_1,relu2_2_2):
	model_def = 'deploy10_2.prototxt'
	#model_weights = 'test.caffemodel'

	net = caffe.Net(model_def,
		            #model_weights,
		            caffe.TEST)

	net.blobs['relu2_2_1'].data[...] = relu2_2_1
	net.blobs['relu2_2_2'].data[...] = relu2_2_2

	net.forward()

	output = net.blobs['pool2_2'].data

	return output
