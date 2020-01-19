import numpy as np
import pylab
import pickle as p
import smc_function as smc
import sys
caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
import caffe

caffe.set_mode_gpu()

def f5(relu1_2):
	model_def = 'deploy5.prototxt'
	#model_weights = 'test.caffemodel'

	net = caffe.Net(model_def,
		            #model_weights,
		            caffe.TEST)

	net.blobs['relu1_2'].data[...] = relu1_2

	net.forward()

	output = net.blobs['pool1'].data

	return output