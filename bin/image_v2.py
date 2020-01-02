#!/usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import logging as log
import numpy as np
import sys
import os

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.datasets import mnist
from keras import optimizers
from keras import layers
from keras import models
from util import *

log.basicConfig(level=log.INFO)

## 设置KERAS根目录，所有的数据集都从这个目录加载
log.info("set ENV `KERAS_HOME` ...")
os.environ["KERAS_HOME"] = "./keras"

## 从`https://www.kaggle.com/c/dogs-vs-cats/data`数据集提取一个小型的数据集
# get_small_dataset("/Users/uc/Downloads/dogs-vs-cats/", "./dataset/dogs-vs-cats")

## 加载图片并标准化
train_datagen = ImageDataGenerator(
	rescale=1./255,
	rotation_range=40,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
		"./dataset/dogs-vs-cats/train",
		# All images will be resized to 150x150
		target_size=(150, 150),
		batch_size=32,
		# Since we use binary_crossentropy loss, we need binary labels
		class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
	   	"./dataset/dogs-vs-cats/validation",
		target_size=(150, 150),
		batch_size=32,
		class_mode='binary')

## 构建模型
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

## 训练模型
history = model.fit_generator(
	train_generator,
	steps_per_epoch = 100,
	epochs = 100,
	validation_data = validation_generator,
	validation_steps = 50)

model.save('./model/cats_and_dogs_small_2.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.subplot(211)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(212)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


