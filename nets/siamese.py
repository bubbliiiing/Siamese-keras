import os

import keras
import keras.backend as K
import numpy as np
from keras.layers import Conv2D, Dense, Flatten, Input, Lambda, MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD
from PIL import Image

from nets.vgg import VGG16


#-------------------------#
#   创建孪生神经网络
#-------------------------#
def siamese(input_shape):
    vgg_model = VGG16()
    
    input_image_1 = Input(shape=input_shape)
    input_image_2 = Input(shape=input_shape)

    #------------------------------------------#
    #   我们将两个输入传入到主干特征提取网络
    #------------------------------------------#
    encoded_image_1 = vgg_model.call(input_image_1)
    encoded_image_2 = vgg_model.call(input_image_2)

    #-------------------------#
    #   相减取绝对值
    #-------------------------#
    l1_distance = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([encoded_image_1, encoded_image_2])

    #-------------------------#
    #   进行两次全连接
    #-------------------------#
    out = Dense(512,activation='relu')(l1_distance)
    #---------------------------------------------#
    #   利用sigmoid函数将最后的值固定在0-1之间。
    #---------------------------------------------#
    out = Dense(1,activation='sigmoid')(out)

    model = Model([input_image_1, input_image_2], out)
    return model
