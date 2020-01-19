## P-CNN: Privacy-Preserving Convolutional neural network model

#### Paper: Edge-Assisted Privacy-preserving Raw Data Sharing Framework for Connected Autonomous Vehicles
文章链接地址：  
&emsp;&emsp;如果您发现本文工作对您有帮助，请考虑引用（参考文献格式）。  
&emsp;&emsp;本文旨在为成熟的大型CNN网络模型（VGG-16）提供安全版本P-CNN，实验表明，P-CNN可以安全正确地实现二分类任务，误差可控制在约1e^-4量级。  
&emsp;&emsp;如果您想复现本文工作，请认真阅读：<https://www.mdeditor.com>;。 

&emsp;&emsp;总结地，包含下述内容：  
&emsp;&emsp;[1. caffe环境配置](#1)  
&emsp;&emsp;[2. 软件配置](#2)  
&emsp;&emsp;[3. 真实数据集](#3)  
&emsp;&emsp;[4. 代码文件功能分布](#4)  
&emsp;&emsp;[5. 训练网络参数](#5)  
&emsp;&emsp;[6. 测试示例](#6)  
&emsp;&emsp;[7. 安全函数介绍](#7)  


### <a id="1">1.环境配置</a>
#### 下载caffe框架源码：  
&emsp;&emsp;下载链接为：[https://github.com/BVLC/caffe](https://github.com/BVLC/caffe); 或者采用clone方式获得，因为本文工作是在caffe框架的基础上进行的，需要调用封装的caffe源码文件；

#### 安装caffe环境通用依赖：
```javascript
sudo apt-get install build-essential -y 
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler  -y 
sudo apt-get install --no-install-recommends libboost-all-dev  -y 
sudo apt-get install libatlas-base-dev  -y 
sudo apt-get install libhdf5-serial-dev -y 
```
#### 修改Makefile.config文件：
```javascript
cd vgg16_secure_caffe/caffe-master
cp Makefile.config.example Makefile.config //或者采用手动复制方式；
```
&emsp;&emsp;取消注释WITH_PYTHON_LAYER := 1，因为caffe需要支持Python layers；

&emsp;&emsp;取消注释USE_CUDNN := 1，因为本文工作实现在GPU模式下；
[注意：如果您需要在CPU模式下复现本文工作，那么应注释USE_CUDNN := 1，而取消注释CPU_ONLY := 1；]

#### 修改hdf5访问地址，将
```javascript
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib
```
&emsp;&emsp;修改为：
```javascript
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial 
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial
```
&emsp;&emsp;同时将Makefile文件中173行：
```javascript
LIBRARIES += glog gflags protobuf boost_system boost_filesystem m hdf5_hl hdf5
```
&emsp;&emsp;修改为：
```javascript
LIBRARIES += glog gflags protobuf boost_system boost_filesystem m hdf5_serial_hl hdf5_serial
```

#### 修改python版本号
```javascript
注释 #PYTHON_INCLUDE:=/usr/include/python2.7 \
/usr/lib/python2.7/dist-packages/numpy/core/include
取消注释 PYTHON_LIBRARIES := boost_python3 python3.5m
PYTHON_INCLUDE := /usr/include/python3.5m \
                             /usr/local/lib/python3.5/dist-packages/numpy/core/include
```
&emsp;&emsp;注意：如果您使用其他版本的python，则需要将3.5改为相应版本号；

#### 编译caffe：
```javascript
cd vgg16_secure_caffe/caffe-master
sudo make clean   #清除配置文件
sudo make all  #编译caffe配置文件
sudo make pycaffe  #编译python接口
```
### <a id="2">2.软件配置</a>
&emsp;&emsp;本文工作使用python3.6版本，需要下载相关的第三方库，主要包括numpy库、time库、opencv库、os库。

### <a id="3">3.真实数据集</a>

&emsp;&emsp;本文工作使用KITTI数据集的一部分进行实验，链接地址为:
<http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark>;，包含3417例训练样本（1001例车类样本，2416例人类样本），750个测试样本（401例车类样本，349例人类样本）。

### <a id="4">4.代码文件功能分布</a>
#### caffe-master文件夹
&emsp;&emsp;包含caffe框架源码文件:
```javascript
cd caffe-master/src/caffe/layers
```
&emsp;&emsp;特别地，其中conv_layer实现网络卷积层功能，relu_layer实现网络激活层功能，pooling_layer实现网络池化层功能，inner_product_layer实现网络全连接层功能。

#### rgb_data文件夹
&emsp;&emsp;存放训练和测试数据，testing为测试图片，包含testing_car和testing_people；train为测试图片，包含train_car和train_people。

#### train文件夹
&emsp;&emsp;包含训练需要的各类文件，其中 ：
- create_train_txt.py负责生成训练样本的类标签，create_val_txt.py负责生成验证样本的类标签，train.txt和val.txt分别为训练样本和验证样本的标签文件；  
- train_val.prototxt为训练网络配置文件，包含13层卷积、5层池化、3层全连接和15层激活；   
- solver.prototxt为训练的设置参数文件，run_zmf.zmf为启动训练的脚本文件。  

#### prototxt类型文件
&emsp;&emsp;配置文件，负责说明特定网络层的名称、输入、输出和类型；  
&emsp;&emsp;deploy1_1、deploy1_2是conv1_1的配置文件；   
&emsp;&emsp;deploy2_1、deploy2_2是relu1_1的配置文件；   
&emsp;&emsp;deploy3_1、deploy3_2是conv1_2的配置文件；     
&emsp;&emsp;deploy4_1、deploy4_2是relu1_2的配置文件；    
&emsp;&emsp;deploy5_1、deploy5_2是pool1的配置文件；    
&emsp;&emsp;deploy6_1、deploy6_2是conv2_1的配置文件；   
&emsp;&emsp;deploy7_1、deploy7_2是relu2_1的配置文件；  
&emsp;&emsp;deploy8_1、deploy8_2是conv2_2的配置文件；   
&emsp;&emsp;deploy9_1、deploy9_2是relu2_2的配置文件；  
&emsp;&emsp;deploy10_1、deploy10_2是pool2的配置文件；  
&emsp;&emsp;deploy11_1、deploy11_2是conv3_1的配置文件；   
&emsp;&emsp;deploy12_1、deploy12_2是relu3_1的配置文件；   
&emsp;&emsp;deploy13_1、deploy13_2是conv3_2的配置文件；   
&emsp;&emsp;deploy14_1、deploy14_2是relu3_2的配置文件；   
&emsp;&emsp;deploy15_1、deploy15_2是conv3_3的配置文件；   
&emsp;&emsp;deploy16_1、deploy16_2是relu3_3的配置文件；   
&emsp;&emsp;deploy17_1、deploy17_2是pool3的配置文件；  
&emsp;&emsp;deploy18_1、deploy18_2是conv4_1的配置文件；   
&emsp;&emsp;deploy19_1、deploy19_2是relu4_1的配置文件；   
&emsp;&emsp;deploy20_1、deploy20_2是conv4_2的配置文件；   
&emsp;&emsp;deploy21_1、deploy21_2是relu4_2的配置文件；   
&emsp;&emsp;deploy22_1、deploy22_2是conv4_3的配置文件；   
&emsp;&emsp;deploy23_1、deploy23_2是relu4_3的配置文件；   
&emsp;&emsp;deploy24_1、deploy24_2是pool4的配置文件；  
&emsp;&emsp;deploy25_1、deploy25_2是conv5_1的配置文件；  
&emsp;&emsp;deploy26_1、deploy26_2是relu5_1的配置文件；  
&emsp;&emsp;deploy27_1、deploy27_2是conv5_2的配置文件；  
&emsp;&emsp;deploy28_1、deploy28_2是relu5_2的配置文件；  
&emsp;&emsp;deploy29_1、deploy29_2是conv5_3的配置文件；  
&emsp;&emsp;deploy30_1、deploy30_2是relu5_3的配置文件；  
&emsp;&emsp;deploy31_1、deploy31_2是pool5的配置文件；  
&emsp;&emsp;deploy32_1、deploy32_2是fc6的配置文件；   
&emsp;&emsp;deploy33_1、deploy33_2是relu6的配置文件；  
&emsp;&emsp;deploy34_1、deploy34_2是fc7的配置文件；  
&emsp;&emsp;deploy35_1、deploy35_2是relu7的配置文件；   
&emsp;&emsp;deploy36_1、deploy36_2是re_fc8的配置文件。  

#### caffemodel类型文件
&emsp;&emsp;训练网络训练出的网络参数；   
&emsp;&emsp;test.caffemodel是本文利用预训练模型迭代10000次训练获得的网络参数文件；  
&emsp;&emsp;revised1是提取conv1_1层的参数；  
&emsp;&emsp;revised2是提取conv1_2层的参数；  
&emsp;&emsp;revised6是提取conv2_1层的参数；  
&emsp;&emsp;revised8是提取conv 2_2层的参数；  
&emsp;&emsp;revised11是提取conv3_1层的参数；  
&emsp;&emsp;revised13是提取conv3_2层的参数；  
&emsp;&emsp;revised15是提取conv3_3层的参数；  
&emsp;&emsp;revised18是提取conv4_1层的参数；  
&emsp;&emsp;revised20是提取conv4_2层的参数；  
&emsp;&emsp;revised22是提取conv4_3层的参数；  
&emsp;&emsp;revised25是提取conv5_1层的参数；  
&emsp;&emsp;revised27是提取conv5_2层的参数；  
&emsp;&emsp;revised29是提取conv5_3层的参数；  
&emsp;&emsp;revised32是提取fc6层的参数；  
&emsp;&emsp;revised34是提取fc7层的参数；  
&emsp;&emsp;revised36是提取re_fc8层的参数。

#### py类型文件
* feature1_1至feature36_2是调用相应配置文件的子函数，用来提供VGG-16各网络层所具备的功能，供feature文件进行调用；  
* layer包含配置文件中定义的python类型的功能函数，即激活relu函数和最大池化max-pool函数；  
* smc_function包含本文设计的各类安全函数SecBitMul、SecBitAdd、SecBitExtra和SecMaxIndex；  
* image_cipher负责生成两张密文图片；  
* read_npy负责读取一些npy类型文件；  
* read_model负责读取test.caffemodel参数文件中部分参数。

#### npy类型文件
&emsp;&emsp;主要是一些在测试图片rgb_data/testing/3.png时获取的各个网络层的特征值矩阵。

### <a id="5">5.训练网络参数</a>
&emsp;&emsp;运行run_zmf.zmf脚本；  
&emsp;&emsp;调用solver.prototxt，供设置各种训练参数，其中VGGnet.caffemodel 为VGG-16网络预训练模型文件，可在:<https://github.com/BVLC/caffe/wiki/Model-Zoomodels-used-by-the-vgg-team-in-ilsvrc-2014>; 处下载（注意，如果您想在CPU模式下进行训练，需要将solver_mode选项改成CPU）；  
&emsp;&emsp;在VGG-16网络配置train_val.prototxt中，我们制作数据集标签文件供训练和验证使用；重要地，我们只训练最后一层全连接层的参数，而不更改其他网络层参数，将其权重和偏置的步长置0；  
```javascript
  param {
    lr_mult: 0
    decay_mult: 0}
```
&emsp;&emsp;注意：如果您更改新的数据集进行训练，若测试效果不理想，可以考虑解放更多的网络层，或者微调训练参数。

### <a id="6">6.测试示例</a>
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

### <a id="7">7.安全函数介绍</a>
#### smc_function.py
* SecBitExtra提供符号位;
* SecBitAdd提供进位加法功能;
* SecBitMul为SecBitAdd提供进位；
* SecMaxIndex提供最大值的二维索引。

#### layer.py
&emsp;&emsp;ReluALayer和ReluBLayer执行relu激活，调用SecBitExtra函数，若bit==1，则意味着像素值小于0，需要将各自像素分量置0;若bit==0，则意味着像素值不小于0，维持各自像素分量不改变。MaxALayer和MaxBLayer调用SecMaxIndex函数，每一个2×2像素矩阵作为输入，返回最大值索引，进而实现max-pool功能
