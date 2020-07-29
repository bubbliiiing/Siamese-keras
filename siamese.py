import keras
import numpy as np
import colorsys
import os
import matplotlib.pyplot as plt
from nets.siamese import siamese
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image,ImageFont, ImageDraw

class Siamese(object):
    _defaults = {
        "model_path": 'model_data/Omniglot_vgg.h5',
        "input_shape" : (105, 105, 3),
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化Siamese
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.sess = K.get_session()
        self.generate()
        
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        self.model = siamese(self.input_shape)
        self.model.load_weights(self.model_path)

        self.model.summary()
        print('{} model, anchors, and classes loaded.'.format(model_path))
    
    def letterbox_image(self, image, size):
        image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        if self.input_shape[-1]==1:
            new_image = new_image.convert("L")
        return new_image

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image_1, image_2):
        image_1 = self.letterbox_image(image_1,[self.input_shape[1],self.input_shape[0]])
        image_2 = self.letterbox_image(image_2,[self.input_shape[1],self.input_shape[0]])
        
        image_1 = np.asarray(image_1).astype(np.float64)/255
        image_2 = np.asarray(image_2).astype(np.float64)/255
        
        if self.input_shape[-1]==1:
            image_1 = np.expand_dims(image_1,-1)
            image_2 = np.expand_dims(image_2,-1)
            
        photo1 = np.expand_dims(image_1,0)
        photo2 = np.expand_dims(image_2,0)

        output = self.model.predict([photo1,photo2])[0]

        plt.subplot(1, 2, 1)
        plt.imshow(np.array(image_1))

        plt.subplot(1, 2, 2)
        plt.imshow(np.array(image_2))
        plt.text(-12, -12, 'Similarity:%.3f' % output, ha='center', va= 'bottom',fontsize=11)
        plt.show()
        return output

    def close_session(self):
        self.sess.close()
