#!/usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import logging as log
import numpy as np
import sys
import os

from keras.utils import to_categorical
from keras.datasets import mnist
from keras import layers
from keras import models

log.basicConfig(level=log.INFO)

## 设置KERAS根目录，所有的数据集都从这个目录加载
log.info("set ENV `KERAS_HOME` ...")
os.environ["KERAS_HOME"] = "./keras"

## 设置GPU参数
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config)) 

## 加载MNIST数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

## 实例化一个小型的卷积神经网络
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

## 在卷积神经网络上添加分类器
model.add(layers.Flatten()) 
model.add(layers.Dense(64, activation='relu')) 
model.add(layers.Dense(10, activation='softmax'))

## 显示CNN的信息
model.summary()

## 训练模型
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) 
model.fit(train_images, train_labels, epochs=5, batch_size=64)

## 测试
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(test_acc)