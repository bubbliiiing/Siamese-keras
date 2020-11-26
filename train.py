from nets.siamese import siamese
from nets.siamese_training import Generator
from nets.siamese_training_own_dataset import Generator as Generator_own_dataset
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam, SGD
import os

def get_image_num(path, train_own_data):
    num = 0
    if train_own_data:
        train_path = os.path.join(path, 'images_background')
        for character in os.listdir(train_path):
            # 在大众类下遍历小种类。
            character_path = os.path.join(train_path, character)
            num += len(os.listdir(character_path))
    else:
        train_path = os.path.join(path, 'images_background')
        for alphabet in os.listdir(train_path):
            # 然后遍历images_background下的每一个文件夹，代表一个大种类
            alphabet_path = os.path.join(train_path, alphabet)
            for character in os.listdir(alphabet_path):
                # 在大众类下遍历小种类。
                character_path = os.path.join(alphabet_path, character)
                num += len(os.listdir(character_path))
    return num

if __name__ == "__main__":
    input_shape = [105,105,3]
    dataset_path = "./datasets"
    log_dir = "logs/"
    # 训练自己的数据
    train_own_data = False

    model = siamese(input_shape)
    model.summary()

    model_path = "model_data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    model.load_weights(model_path, by_name=True, skip_mismatch=True)
    # 保存的方式，3世代保存一次
    checkpoint_period = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    # tensorboard
    tensorboard = TensorBoard(log_dir=log_dir)

    train_ratio = 0.9
    images_num = get_image_num(dataset_path, train_own_data)
    train_num = int(images_num*0.9)
    val_num = int(images_num*0.1)
    
    if True:
        # 交叉熵
        Batch_size = 32
        Lr = 1e-3
        Init_epoch = 0
        Freeze_epoch = 25
        
        model.compile(loss = "binary_crossentropy",
                    optimizer = Adam(lr=Lr),
                    metrics = ["binary_accuracy"])
        print('Train with batch size {}.'.format(Batch_size))

        if train_own_data:
            gen = Generator_own_dataset(input_shape, dataset_path, Batch_size, train_ratio)
        else:
            gen = Generator(input_shape, dataset_path, Batch_size, train_ratio)
            
        # 开始训练
        model.fit_generator(gen.generate(True),
                steps_per_epoch=max(1,train_num//Batch_size),
                validation_data=gen.generate(False),
                validation_steps=max(1,val_num//Batch_size),
                epochs=Freeze_epoch,
                initial_epoch=Init_epoch,
                callbacks=[checkpoint_period, reduce_lr, early_stopping, tensorboard])

    if True:
        # 交叉熵
        Batch_size = 32
        Lr = 1e-4
        Freeze_epoch = 25
        Epoch = 50
        
        model.compile(loss = "binary_crossentropy",
                    optimizer = Adam(lr=Lr),
                    metrics = ["binary_accuracy"])
        print('Train with batch size {}.'.format(Batch_size))

        if train_own_data:
            gen = Generator_own_dataset(input_shape, dataset_path, Batch_size, train_ratio)
        else:
            gen = Generator(input_shape, dataset_path, Batch_size, train_ratio)
        # 开始训练
        model.fit_generator(gen.generate(True),
                steps_per_epoch=max(1,train_num//Batch_size),
                validation_data=gen.generate(False),
                validation_steps=max(1,val_num//Batch_size),
                epochs=Epoch,
                initial_epoch=Freeze_epoch,
                callbacks=[checkpoint_period, reduce_lr, early_stopping, tensorboard])
