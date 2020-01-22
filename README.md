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
* smc_function_number.py is the number vision of secure function proposed in this work.    
* smc_function_matrix.py is the Numpy matrix vision of secure function proposed in this work. 
* read_npy is used to read the .npy type files.  
* read_model is used to read the parameter in .caffemodel type files.

#### .npy files.
&emsp;&emsp; Numpy matrix data, e.g., network parameter.

### <a id="5">5. Training network parameters</a>
&emsp;&emsp; operate run_zmf.zmf script.  
&emsp;&emsp; solver.prototxt is used to set training parameters.
&emsp;&emsp; VGGnet.caffemodel is pre-training model file in VGG16 network.download link: <https://github.com/BVLC/caffe/wiki/Model-Zoomodels-used-by-the-vgg-team-in-ilsvrc-2014>; 
&emsp;&emsp; Note that: If you want to train in CPU mode, you need to change the solver_mode option to CPU.
&emsp;&emsp; In train_val.prototxt, we make the dataset label file for training and verification.
&emsp;&emsp; Specially, we only train the parameters of the last full-connected layer without changing the parameters of the other network layers, setting the offset of weight and bias to 0.
```javascript
  param {
    lr_mult: 0
    decay_mult: 0}
```
&emsp;&emsp; Note that: if you change a new dataset for training, consider freeing up more network layers or fine-tuning training parameters if the test results are not ideal.

### <a id="6">6. Test demo</a>
&emsp;&emsp; Run feature.py.  
&emsp;&emsp; Test sample: data/testing/3.png.  
```javascript
# Modify Caffe path correctly, please.
caffe_root = '/home/hadoop/workspace/caffe-master'
sys.path.append('/home/hadoop/workspace/caffe-master/python')
```
&emsp;&emsp; Note that: If you want to perform the test process in CPU mode, you need modify caffe.set_mode_gpu() as caffe.set_mode_cpu().

#### Mainly includes the following process:
* Image encryption: read image, crop the image size into 224×224×3, transpose the image matrix to 3×224×224, split randomly image to       image1 and image2 with the same dimension.
* Feature extraction and classification: perform the secure functions proposed in this work according to the order of vgg-16 network,     and output the classification componets, i.e., score1 and score2.
* Image decryption: According to the idea of addition secret sharing, two scores (i.e., score_inter) can be obtained by samply addition, and the class label corresponding to the larger score is denoted as final classification result.

### <a id="7">7. Introduction of secure function</a>
#### smc_function_matrix.py
* SecBitExtra is used to provide symbol-bit matrix of input matrix;
* SecBitAdd is used to realize the secure carry addition;
* SecBitMul is used to provide carry for SecBitAdd;
* SecMaxIndex is used to obtain the maximum location in pooling area.
