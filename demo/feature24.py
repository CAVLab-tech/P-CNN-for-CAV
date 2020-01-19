import numpy as np
import pylab
import pickle as p
import smc_function as smc
import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()

def f24(relu4_3):
	model_def = 'deploy24.prototxt'
	#model_weights = 'test.caffemodel'

	net = caffe.Net(model_def,
		            #model_weights,
		            caffe.TEST)

	net.blobs['relu4_3'].data[...] = relu4_3

	net.forward()

	output = net.blobs['pool4'].data

	return output