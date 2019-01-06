# set the matplotlib backend so figures can be saved in the background
# import matplotlib
# matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import sys  # 包含了与python解释器和它的环境有关的函数

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K


sys.path.append('..')  # 将模块路径加到当前模块扫描的路径里
from net.lenet import LeNet



# initialize the number of epochs to train for
# and batch size
EPOCHS = 20
INIT_LR = 0.3e-3  # initial learning rate
BS = 50  # Batch Size
CLASS_NUM = 6  # 分类数
norm_size = 100  # 归一化大小


def load_data(path):
    print("[INFO] loading images...")
    data = []
    labels = []
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(path)))  # 排序，转化为列表
    random.seed(42)
    random.shuffle(imagePaths)
    # loop over the input images
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (norm_size, norm_size))
        image = img_to_array(image)
        data.append(image)  # append() 方法向列表的尾部添加一个新的元素。

        # extract the class label from the image path and update the
        # labels list
        label = int(imagePath.split(os.path.sep)[-2])
        # 通过指定分隔符(路径分隔符)对字符串进行切片，右数第二个元素，即文件夹名
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # convert the labels from integers to vectors
    labels = to_categorical(labels, num_classes=CLASS_NUM)
    # 转化为类向量
    return data, labels


def train(aug, trainX, trainY, testX, testY):
    # initialize the model
    print("[INFO] compiling model...")

    base_model = InceptionV3(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # 增加全连接层
    x = Dense(1000, activation='relu')(x)
    # softmax激活函数用户分类
    predictions = Dense(6, activation='softmax')(x)

    # 预训练模型与新加层的组合
    model = Model(inputs=base_model.input, outputs=predictions)

    # 只训练新加的Top层，冻结InceptionV3所有层
    layernum=0

  #  for layer in base_model.layers:
   #     layernum=layernum+1
  #      layer.trainable = True

  #  print(layernum)

    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=["accuracy"])


#optimizer=opt,optimizer='rmsprop'

    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                            validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                            epochs=EPOCHS, verbose=1)

    # save the model to disk
    print("[INFO] serializing network...")
#    model.save('H:\\Turbine\\alltran.model')

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure("loss")
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    # plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    # plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss ")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.show()

    plt.style.use("ggplot")
    plt.figure("acc")
    N = EPOCHS
    # plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    # plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")
    plt.show()


# python train.py --dataset_train ../../traffic-sign/train --dataset_test ../../traffic-sign/test --model traffic_sign.model
if __name__ == '__main__':
    train_file_path = "H:\\Turbine\\Train"
    test_file_path = "H:\\Turbine\\Test"
    trainX, trainY = load_data(train_file_path)
    testX, testY = load_data(test_file_path)
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=360, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, vertical_flip=True, fill_mode="nearest")
    # 旋转角度,水平垂直位移量，逆时针方向剪切角，随机缩放的幅度，进行随机水平翻转，进行变换时超出边界的点将根据本参数给定的方法进行处理
    train(aug, trainX, trainY, testX, testY)