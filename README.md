## P-CNN: Privacy-Preserving Convolutional Neural Network

#### Paper: Edge-Assisted Privacy-preserving Raw Data Sharing Framework for Connected Autonomous Vehicles
The paper is available at <https://×××.com>.

&emsp;&emsp;If you find the work of this paper helpful, consider quoting it (bibtex).  
&emsp;&emsp;This paper aims to provide a secure Numpy matrix version of P-CNN for the mature large CNN network model (e.g., vgg-16). Experiments show that P-CNN can securely and correctly implement the task of binary classification, and the error can be controlled within 1e-4.
&emsp;&emsp;If you want to reproduce the work of this paper, please read the code in <https://github.com/CAVLab-tech/P-CNN-for-CAV>.

&emsp;&emsp;In summary, the code includes the following:  
&emsp;&emsp;[1. Caffe environment configuration](#1)  
&emsp;&emsp;[2. Software configuration](#2)  
&emsp;&emsp;[3. Real dataset](#3)  
&emsp;&emsp;[4. Code file function distribution](#4)  
&emsp;&emsp;[5. Training network parameters](#5)  
&emsp;&emsp;[6. Test demo](#6)  
&emsp;&emsp;[7. Introduction of secure function](#7)  


### <a id="1">1. Caffe environment configuration</a>
#### Download Caffe framework source code:  
&emsp;&emsp;Download link: [https://github.com/BVLC/caffe](https://github.com/BVLC/caffe), or download using clone mode. 
Because this experiment is based on the Caffe framework, we need to call the encapsulated Caffe source file.

#### Install Caffe environment universal dependency:
```javascript
sudo apt-get install build-essential -y 
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler  -y 
sudo apt-get install --no-install-recommends libboost-all-dev  -y 
sudo apt-get install libatlas-base-dev  -y 
sudo apt-get install libhdf5-serial-dev -y 
```
#### Modify makefile.config file:
```javascript
cd vgg16_secure_caffe/caffe-master
cp Makefile.config.example Makefile.config  //Or copy manually.
```
&emsp;&emsp;Uncomment WITH_PYTHON_LAYER := 1, because caffe framwork needs to support Python layers.

&emsp;&emsp;Uncomment USE_CUDNN := 1, because this works in GPU mode.
[Note that: If you need to replicate the work in CPU mode, you should comment USE_CUDNN := 1, and uncomment CPU_ONLY := 1.]

#### Modify the hdf5 access address: 
&emsp;&emsp; Modify
```javascript
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib
```
&emsp;&emsp; as
```javascript
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial 
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial
```
&emsp;&emsp; Modify (line 173 in Makefile file)
```javascript
LIBRARIES += glog gflags protobuf boost_system boost_filesystem m hdf5_hl hdf5
```
&emsp;&emsp; as
```javascript
LIBRARIES += glog gflags protobuf boost_system boost_filesystem m hdf5_serial_hl hdf5_serial
```

#### Modify the version number of Python:
```javascript
Comment #PYTHON_INCLUDE:=/usr/include/python2.7 \
/usr/lib/python2.7/dist-packages/numpy/core/include
Uncomment PYTHON_LIBRARIES := boost_python3 python3.5m
PYTHON_INCLUDE := /usr/include/python3.5m \
                             /usr/local/lib/python3.5/dist-packages/numpy/core/include
```
&emsp;&emsp; Note that: If you are using another version of Python, you need to change 3.5 to the corresponding version number.

#### Compile caffe:
```javascript
cd vgg16_secure_caffe/caffe-master
sudo make clean   #Clear configuration file.
sudo make all  # Compile caffe configuration file.
sudo make pycaffe  #Compile python Interface.
```
### <a id="2">2. Software configuration</a>
&emsp;&emsp; This work with Python 3.5 and requires the download of relevant third-party libraries, including Numpy library, Time library, Opencv library, and OS library.

### <a id="3">3. Real dataset</a>

&emsp;&emsp;  This work process the KITTI dataset <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark> by cropping out the vehicles and pedestrians from each frame provided in the dataset. 
It includes 3,417 training samples (1,001 vehicle samples and 2,416 human samples) and 750 test samples (401 vehicle samples and 349 human samples).

### <a id="4">4. Code file function distribution</a>
#### caffe-master folder.
&emsp;&emsp; Contain caffe framework source file:
```javascript
cd caffe-master/src/caffe/layers
```
&emsp;&emsp; In particular, conv_layer file realizes the function of network convolution layer, relu_layer file realizes the function of network activation layer, pooling_layer file realizes the function of network pooling layer, and inner_product_layer file realizes the function of network full connection layer.

#### data folder.
&emsp;&emsp; Store training and test data. Testing file stores test image, including testing_car and testing_people. Train file stores training image, including train_car and train_people.

#### train folder.
&emsp;&emsp; Contain various files required for training, including:
- create_train_txt.py is responsible for generating class labels for training samples.
  create_val_txt.py is responsible for generating class labels for verification samples.
  
- train_val.prototxt is the training network configuration file, including 13 convolution layers, 5 pooling layers, 3 full-connected       layers and 15 activation layers.  
- solver.prototxt is the Settings file for the training, and run_zmf.zmf is the script file to start the training.

#### prototxt files.
&emsp;&emsp;  The configuration file that specifies the name, input, output, and type of a particular network layer;
&emsp;&emsp;deploy1_1 and deploy1_2 are the configuration file of conv1_1;
&emsp;&emsp;deploy2_1 and deploy2_2 are the configuration file of relu1_1; 
&emsp;&emsp;deploy3_1 and deploy3_2 are the configuration file of conv1_2;     
&emsp;&emsp;deploy4_1 and deploy4_2 are the configuration file of relu1_2;    
&emsp;&emsp;deploy5_1 and deploy5_2 are the configuration file of pool1;    
&emsp;&emsp;deploy6_1 and deploy6_2 are the configuration file of conv2_1;   
&emsp;&emsp;deploy7_1 and deploy7_2 are the configuration file of relu2_1;  
&emsp;&emsp;deploy8_1 and deploy8_2 are the configuration file of conv2_2;   
&emsp;&emsp;deploy9_1 and deploy9_2 are the configuration file of relu2_2;  
&emsp;&emsp;deploy10_1 and deploy10_2 are the configuration file of pool2;  
&emsp;&emsp;deploy11_1 and deploy11_2 are the configuration file of conv3_1;   
&emsp;&emsp;deploy12_1 and deploy12_2 are the configuration file of relu3_1;   
&emsp;&emsp;deploy13_1 and deploy13_2 are the configuration file of conv3_2;   
&emsp;&emsp;deploy14_1 and deploy14_2 are the configuration file of relu3_2;   
&emsp;&emsp;deploy15_1 and deploy15_2 are the configuration file of conv3_3;   
&emsp;&emsp;deploy16_1 and deploy16_2 are the configuration file of relu3_3;   
&emsp;&emsp;deploy17_1 and deploy17_2 are the configuration file of pool3; 
&emsp;&emsp;deploy18_1 and deploy18_2 are the configuration file of conv4_1;   
&emsp;&emsp;deploy19_1 and deploy19_2 are the configuration file of relu4_1;   
&emsp;&emsp;deploy20_1 and deploy20_2 are the configuration file of conv4_2;   
&emsp;&emsp;deploy21_1 and deploy21_2 are the configuration file of relu4_2;   
&emsp;&emsp;deploy22_1 and deploy22_2 are the configuration file of conv4_3;   
&emsp;&emsp;deploy23_1 and deploy23_2 are the configuration file of relu4_3;   
&emsp;&emsp;deploy24_1 and deploy24_2 are the configuration file of pool4;  
&emsp;&emsp;deploy25_1 and deploy25_2 are the configuration file of conv5_1;  
&emsp;&emsp;deploy26_1 and deploy26_2 are the configuration file of relu5_1;  
&emsp;&emsp;deploy27_1 and deploy27_2 are the configuration file of conv5_2;  
&emsp;&emsp;deploy28_1 and deploy28_2 are the configuration file of relu5_2;  
&emsp;&emsp;deploy29_1 and deploy29_2 are the configuration file of conv5_3;  
&emsp;&emsp;deploy30_1 and deploy30_2 are the configuration file of relu5_3;  
&emsp;&emsp;deploy31_1 and deploy31_2 are the configuration file of pool5;  
&emsp;&emsp;deploy32_1 and deploy32_2 are the configuration file of fc6;   
&emsp;&emsp;deploy33_1 and deploy33_2 are the configuration file of relu6;  
&emsp;&emsp;deploy34_1 and deploy34_2 are the configuration file of fc7; 
&emsp;&emsp;deploy35_1 and deploy35_2 are the configuration file of relu7;  
&emsp;&emsp;deploy36_1 and deploy36_2 are the configuration file of re_fc8.  

#### caffemodel files.
&emsp;&emsp; The network parameters trained by the training network.   
&emsp;&emsp;test.caffemodel is the network parameter file obtained by iterating 10,000 times through the pre-training model .
&emsp;&emsp;revised1 is the parameter of extracting conv1_1 layer;  
&emsp;&emsp;revised2 is the parameter of extracting conv1_2 layer;  
&emsp;&emsp;revised6 is the parameter of extracting conv2_1 layer;   
&emsp;&emsp;revised8 is the parameter of extracting conv2_2 layer;   
&emsp;&emsp;revised11 is the parameter of extracting conv3_1 layer;  
&emsp;&emsp;revised13 is the parameter of extracting conv3_2 layer;  
&emsp;&emsp;revised15 is the parameter of extracting conv3_3 layer;   
&emsp;&emsp;revised18 is the parameter of extracting conv4_1 layer;   
&emsp;&emsp;revised20 is the parameter of extracting conv4_2 layer;    
&emsp;&emsp;revised22 is the parameter of extracting conv4_3 layer;   
&emsp;&emsp;revised25 is the parameter of extracting conv5_1 layer;   
&emsp;&emsp;revised27 is the parameter of extracting conv5_2 layer;   
&emsp;&emsp;revised29 is the parameter of extracting conv5_3 layer;   
&emsp;&emsp;revised32 is the parameter of extracting fc6 layer;    
&emsp;&emsp;revised34 is the parameter of extracting fc7 layer;   
&emsp;&emsp;revised36 is the parameter of extracting re_fc8 layer.

#### python files.
* feature1_1.py to feature36_2.py are the sub-functions that call the corresponding configuration files, and provide normal function of   each layer in VGG16.
* layer.py contains the ReLU function and the Max-pool function.
* smc_function.py 包含本文设计的各类安全函数SecBitMul、SecBitAdd、SecBitExtra和SecMaxIndex；  
* image_cipher负责生成两张密文图片；  
* read_npy负责读取一些npy类型文件；  
* read_model负责读取test.caffemodel参数文件中部分参数。

#### npy类型文件
&emsp;&emsp;主要是一些在测试图片rgb_data/testing/3.png时获取的各个网络层的特征值矩阵。

### <a id="5">5. Training network parameters</a>
&emsp;&emsp;运行run_zmf.zmf脚本；  
&emsp;&emsp;调用solver.prototxt，供设置各种训练参数，其中VGGnet.caffemodel 为VGG-16网络预训练模型文件，可在:<https://github.com/BVLC/caffe/wiki/Model-Zoomodels-used-by-the-vgg-team-in-ilsvrc-2014>; 处下载（注意，如果您想在CPU模式下进行训练，需要将solver_mode选项改成CPU）；  
&emsp;&emsp;在VGG-16网络配置train_val.prototxt中，我们制作数据集标签文件供训练和验证使用；重要地，我们只训练最后一层全连接层的参数，而不更改其他网络层参数，将其权重和偏置的步长置0；  
```javascript
  param {
    lr_mult: 0
    decay_mult: 0}
```
&emsp;&emsp;注意：如果您更改新的数据集进行训练，若测试效果不理想，可以考虑解放更多的网络层，或者微调训练参数。

### <a id="6">6. Test demo</a>
&emsp;&emsp;运行feature.py  
&emsp;&emsp;测试示例为rgb_data/testing/3.png  
```javascript
# 正确更改caffe路径
caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
```
&emsp;&emsp;注意：如果您想在CPU模式下进行测试，则需要将caffe.set_mode_gpu()更改为caffe.set_mode_cpu()；
#### 主要包含下述过程：
* 图片加密：读入图片、crop图片维度为224×224×3、像素矩阵转置为3×224×224、随机分割图片image为image1和image2；
* 特征提取：依据VGG-16网络顺序调用各个网络层的功能函数，获得分割的特征分量score1和score2；  
注意：依据和，我们对于第二个网络输出需要减去偏置bias，满足实验要求；
* 图片解密：根据加法秘密共享性质，可获得测试示例的两个特征分数score_inter，较大分数对应的类标签记为示例测试输出。

### <a id="7">7. Introduction of secure function</a>
#### smc_function.py
* SecBitExtra提供符号位;
* SecBitAdd提供进位加法功能;
* SecBitMul为SecBitAdd提供进位；
* SecMaxIndex提供最大值的二维索引。

#### layer.py
&emsp;&emsp;ReluALayer和ReluBLayer执行relu激活，调用SecBitExtra函数，若bit==1，则意味着像素值小于0，需要将各自像素分量置0;若bit==0，则意味着像素值不小于0，维持各自像素分量不改变。MaxALayer和MaxBLayer调用SecMaxIndex函数，每一个2×2像素矩阵作为输入，返回最大值索引，进而实现max-pool功能
