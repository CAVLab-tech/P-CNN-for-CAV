
import numpy as np
import sys


from smc_function import smc_functions

caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')

import caffe

class ReluALayer(caffe.Layer):

	def setup(self, bottom, top):
		# check input pair
		pass

	def reshape(self, bottom, top):
		# top[0].reshape(*bottom[0].data.shape)
		# copy shape from bottom:
		top[0].reshape(bottom[0].num, bottom[0].channels, bottom[0].height, bottom[0].width)

	def forward(self, bottom, top):

		smc = smc_functions()
		bn = bottom[0].num
		bc = bottom[0].channels
		bh = bottom[0].height
		bw = bottom[0].width
		for n in range(bn):
			for c in range(bc):
				for h in range(bh):
					for w in range(bw):
						bit_1,bit_2 = smc.SecBitExtra(bottom[0].data[n,c,h,w], bottom[1].data[n,c,h,w])
						bit= bit_1^bit_2
						if bit ==1:
							bottom[0].data[n, c, h, w] =0
		top[0].data[...] = bottom[0].data


	def backward(self, top, propagate_down, bottom):
		pass


class ReluBLayer(caffe.Layer):

	def setup(self, bottom, top):
		pass

	def reshape(self, bottom, top):
		top[0].reshape(bottom[0].num, bottom[0].channels, bottom[0].height, bottom[0].width)
		

	def forward(self, bottom, top):

		smc = smc_functions()
		bn = bottom[0].num
		bc = bottom[0].channels
		bh = bottom[0].height
		bw = bottom[0].width
		for n in range(bn):
			for c in range(bc):
				for h in range(bh):
					for w in range(bw):
						bit_1, bit_2 = smc.SecBitExtra(bottom[0].data[n, c, h, w], bottom[1].data[n, c, h, w])
						bit= bit_1^bit_2
						if bit == 1:
							bottom[1].data[n, c, h, w] = 0
		top[0].data[...] = bottom[1].data


	def backward(self, top, propagate_down, bottom):
		pass

class ReluELayer(caffe.Layer):

	def setup(self, bottom, top):
		# check input pair
		pass

	def reshape(self, bottom, top):
		top[0].reshape(bottom[0].num, bottom[0].channels, bottom[0].height, bottom[0].width)

	def forward(self, bottom, top):

		top[0].data[...] = bottom[0].data
		temp = bottom[0].data
		top[0].data[temp < 0] = 0

	def backward(self, top, propagate_down, bottom):
		pass


class MaxALayer(caffe.Layer):
	def setup(self, bottom, top):
		pass

	def reshape(self, bottom, top):
		top[0].reshape(bottom[0].num, bottom[0].channels, int(bottom[0].height/2), int(bottom[0].width/2))
		

	def forward(self, bottom, top):
		bn = bottom[0].num
		bc = bottom[0].channels
		bh = bottom[0].height
		bw = bottom[0].width
		tn = bn
		tc = bc
		th = int(bh / 2)
		tw = int(bw / 2)

		smc = smc_functions()

		for n in range(tn):
			for c in range(tc):

				temp_b1 = bottom[0].data[n, c]
				temp_b2 = bottom[1].data[n, c]
				temp_b = temp_b1 + temp_b2

				temp_t1 = np.zeros((th, tw))
				for h in range(th):
					for w in range(tw):

						x,y = smc.SecMaxIndex(temp_b1[h*2:h*2+2, w*2:w*2+2],temp_b2[h*2:h*2+2, w*2:w*2+2])
						temp_t1[h,w]=temp_b1[h*2+x,w*2+y]

				top[0].data[n, c] = temp_t1
				

class MaxBLayer(caffe.Layer):
	def setup(self, bottom, top):
		pass

	def reshape(self, bottom, top):
		top[0].reshape(bottom[0].num, bottom[0].channels, int(bottom[0].height/2), int(bottom[0].width/2))
		

	def forward(self, bottom, top):
		bn = bottom[0].num
		bc = bottom[0].channels
		bh = bottom[0].height
		bw = bottom[0].width
		tn = bn
		tc = bc
		th = int(bh / 2)
		tw = int(bw / 2)

		smc = smc_functions()

		for n in range(tn):
			for c in range(tc):

				temp_b1 = bottom[0].data[n, c]
				temp_b2 = bottom[1].data[n, c]
				temp_b = temp_b1 + temp_b2

				temp_t2 = np.zeros((th, tw))
				for h in range(th):
					for w in range(tw):

						x, y = smc.SecMaxIndex(temp_b1[h * 2:h * 2 + 2, w * 2:w * 2 + 2],temp_b2[h * 2:h * 2 + 2, w * 2:w * 2 + 2])
						temp_t2[h, w] = temp_b2[h*2+x, w*2+y]

				top[0].data[n, c] = temp_t2
				

	def backward(self, top, propagate_down, bottom):
		pass


class MaxCLayer(caffe.Layer):
	def setup(self, bottom, top):
		pass

	def reshape(self, bottom, top):
		top[0].reshape(bottom[0].num, bottom[0].channels, int(bottom[0].height / 2), int(bottom[0].width / 2))

	def forward(self, bottom, top):
		bn = bottom[0].num
		bc = bottom[0].channels
		bh = bottom[0].height
		bw = bottom[0].width
		tn = bn
		tc = bc
		th = int(bh / 2)
		tw = int(bw / 2)

		for n in range(tn):
			for c in range(tc):
				temp_b = bottom[0].data[n, c]
				temp_t = np.zeros((th, tw))

				for h in range(th):
					for w in range(tw):
						position = np.where(temp_b == np.max(temp_b[h*2:h*2+2, w*2:w*2+2]))
						p_x = position[0][0]
						p_y = position[1][0]
						temp_t[h, w] = temp_b[p_x, p_y]

				top[0].data[n, c] = temp_t
