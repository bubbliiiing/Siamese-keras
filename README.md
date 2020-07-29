## Siamese：孪生神经网络在Keras当中的实现
---

### 目录
1. [实现的内容 Achievement](#实现的内容)
2. [所需环境 Environment](#所需环境)
3. [注意事项 Attention](#注意事项)
4. [文件下载 Download](#文件下载)
5. [预测步骤 How2predict](#预测步骤)
6. [训练步骤 How2train](#训练步骤)
7. [参考资料 Reference](#Reference)

### 实现的内容
该仓库实现了孪生神经网络（Siamese network），该网络常常用于检测输入进来的两张图片的相似性。该仓库所使用的主干特征提取网络（backbone）为VGG16。

### 所需环境
tensorflow-gpu==1.13.1  
keras==2.1.5  

### 注意事项
**训练Omniglot数据集和训练自己的数据集可以采用两种不同的格式**。需要注意格式的摆放噢！

### 文件下载
训练所需的vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5可在百度网盘中下载。  
链接: https://pan.baidu.com/s/1FF79PmRc8BzZk8M_ARdMmw 提取码: dc2j  
我一共会提供两个权重，分别是vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5和Omniglot_vgg.h5。
其中:
Omniglot_vgg.h5是Omniglot训练好的权重，可直接使用进行下面的预测步骤。
vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5是vgg的权重，可以用于训练其它的数据集。

### 预测步骤
#### 1、使用预训练权重
下载完库后解压，在百度网盘下载Omniglot_vgg.h5，放入model_data，运行predict.py，依次输入  
```python
img/street.jpg
```
可完成预测。  
#### 2、使用自己训练的权重
a、按照训练步骤训练。  
b、在siamese.py文件里面，在如下部分修改model_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件**。  
```python
_defaults = {
    "model_path": 'model_data/Omniglot_vgg.h5',
    "input_shape" : (105, 105, 3),
}
```
c、运行predict.py，输入  
```python
img/street.jpg
```
可完成预测。  
d、利用video.py可进行摄像头检测。  

### 训练步骤
## 1、训练本文所使用的Omniglot例子
![1](https://img-blog.csdnimg.cn/20200714212548476.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70)
下载数据集，放在根目录下的dataset文件夹下。   
![2](https://img-blog.csdnimg.cn/20200714212649786.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70)
运行train.py开始训练。  
![3](https://img-blog.csdnimg.cn/20200714212953877.png)
## 2、训练自己相似性比较的模型
如果大家想要训练自己的数据集，可以将数据集按照如下格式进行摆放。   
![4](https://img-blog.csdnimg.cn/20200717132416288.png)
每一个chapter里面放同类型的图片。   
之后将train.py当中的train_own_data设置成True，即可开始训练。   
![5](https://img-blog.csdnimg.cn/20200717132625692.png)

### Reference
https://github.com/qqwweee/keras-yolo3/  
https://github.com/Cartucho/mAP  
https://github.com/Ma-Dan/keras-yolo4  
