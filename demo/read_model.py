import numpy as np
import pylab
import pickle as p
import time
import sys

# caffe_root = '/home/cqx/caffe/'
# sys.path.append('/home/cqx/caffe/python')

caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')

import caffe  
caffe.set_mode_gpu()

model_def = 'deploy36.prototxt'
model_weights = 'test.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)

# conv1_1_w=net.params['conv1_1'][0].data
# conv1_1_b=net.params['conv1_1'][1].data
# np.save('conv1_1_b.npy',conv1_1_b)
# net.save('/home/hadoop/workspace/caffe-master/revised1.caffemodel')

# conv1_2_w=net.params['conv1_2'][0].data
# conv1_2_b=net.params['conv1_2'][1].data
# np.save('conv1_2_b.npy',conv1_2_b)
# net.save('/home/hadoop/workspace/caffe-master/revised3.caffemodel')

# conv2_1_w=net.params['conv2_1'][0].data
# conv2_1_b=net.params['conv2_1'][1].data
# np.save('conv2_1_b.npy',conv2_1_b)
# net.save('/home/hadoop/workspace/caffe-master/revised6.caffemodel')

# conv2_2_w=net.params['conv2_2'][0].data
# conv2_2_b=net.params['conv2_2'][1].data
# np.save('conv2_2_b.npy',conv2_2_b)
# net.save('/home/hadoop/workspace/caffe-master/revised8.caffemodel')

# conv3_1_w=net.params['conv3_1'][0].data
# conv3_1_b=net.params['conv3_1'][1].data
# np.save('conv3_1_b.npy',conv3_1_b)
# net.save('/home/hadoop/workspace/caffe-master/revised11.caffemodel')

# conv3_2_w=net.params['conv3_2'][0].data
# conv3_2_b=net.params['conv3_2'][1].data
# np.save('conv3_2_b.npy',conv3_2_b)
# net.save('/home/hadoop/workspace/caffe-master/revised13.caffemodel')

# conv3_3_w=net.params['conv3_3'][0].data
# conv3_3_b=net.params['conv3_3'][1].data
# np.save('conv3_3_b.npy',conv3_3_b)
# net.save('/home/hadoop/workspace/caffe-master/revised15.caffemodel')

# conv4_1_w=net.params['conv4_1'][0].data
# conv4_1_b=net.params['conv4_1'][1].data
# np.save('conv4_1_b.npy',conv4_1_b)
# net.save('/home/hadoop/workspace/caffe-master/revised18.caffemodel')

# conv4_2_w=net.params['conv4_2'][0].data
# conv4_2_b=net.params['conv4_2'][1].data
# np.save('conv4_2_b.npy',conv4_2_b)
# net.save('/home/hadoop/workspace/caffe-master/revised20.caffemodel')

# conv4_3_w=net.params['conv4_3'][0].data
# conv4_3_b=net.params['conv4_3'][1].data
# np.save('conv4_3_b.npy',conv4_3_b)
# net.save('/home/hadoop/workspace/caffe-master/revised22.caffemodel')

# conv5_1_w=net.params['conv5_1'][0].data
# conv5_1_b=net.params['conv5_1'][1].data
# np.save('conv5_1_b.npy',conv5_1_b)
# net.save('/home/hadoop/workspace/caffe-master/revised25.caffemodel')

# conv5_2_w=net.params['conv5_2'][0].data
# conv5_2_b=net.params['conv5_2'][1].data
# np.save('conv5_2_b.npy',conv5_2_b)
# net.save('/home/hadoop/workspace/caffe-master/revised27.caffemodel')

# conv5_3_w=net.params['conv5_3'][0].data
# conv5_3_b=net.params['conv5_3'][1].data
# np.save('conv5_3_b.npy',conv5_3_b)
# net.save('/home/hadoop/workspace/caffe-master/revised29.caffemodel')

# fc6_w=net.params['fc6'][0].data
# fc6_b=net.params['fc6'][1].data
# np.save('fc6_b.npy',fc6_b)
# net.save('/home/hadoop/workspace/caffe-master/revised32.caffemodel')

# fc7_w=net.params['fc7'][0].data
# fc7_b=net.params['fc7'][1].data
# np.save('fc7_b.npy',fc7_b)
# net.save('/home/hadoop/workspace/caffe-master/revised34.caffemodel')

score_w=net.params['re_fc8'][0].data
score_b=net.params['re_fc8'][1].data
np.save('score_b.npy',score_b)
#net.save('/home/hadoop/workspace/caffe-master/revised36.caffemodel')



#net.params['conv1_1'][1].data =np.zeros((1,64))    #revise
# print(conv1_w.size, conv1_b.size)
# print(re_fc8_w.size)