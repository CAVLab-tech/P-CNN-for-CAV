import numpy as np
import pylab
import pickle as p
import smc_function as smc
import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()

def f24_1(relu4_3_1, relu4_3_2):
	model_def = 'deploy24_1.prototxt'
	#model_weights = 'test.caffemodel'

	net = caffe.Net(model_def,
		            #model_weights,
		            caffe.TEST)

	net.blobs['relu4_3_1'].data[...] = relu4_3_1
	net.blobs['relu4_3_2'].data[...] = relu4_3_2
	net.forward()

	output = net.blobs['pool4_1'].data

	return output
