#!/usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import logging as log
import numpy as np
import sys
import os

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import optimizers
from keras import models 
from keras import layers 

log.basicConfig(level=log.INFO)

## 设置KERAS根目录，所有的数据集都从这个目录加载
log.info("set ENV `KERAS_HOME` ...")
os.environ["KERAS_HOME"] = "./keras"

## 设置GPU参数
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config=config)) 

## 加载预训练的VGG16模型
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
conv_base.summary()

## 加载图片
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
	features = np.zeros(shape=(sample_count, 4, 4, 512))
	labels = np.zeros(shape=(sample_count))
	print("A")
	generator = datagen.flow_from_directory(
		directory,
		target_size=(150, 150),
		batch_size=batch_size,
		class_mode='binary')
	print("B")
	i = 0
	for inputs_batch, labels_batch in generator:
		features_batch = conv_base.predict(inputs_batch)
		features[i * batch_size : (i + 1) * batch_size] = features_batch
		labels[i * batch_size : (i + 1) * batch_size] = labels_batch
		i += 1
		if i % 10 == 0:
			print("process ", i , " images...")
		if i * batch_size >= sample_count:
			# Note that since generators yield data indefinitely in a loop,
			# we must `break` after every image has been seen once.
			break
	return features, labels

## 提取特征
train_features, train_labels = extract_features("./dataset/dogs-vs-cats/train", 2000)
validation_features, validation_labels = extract_features("./dataset/dogs-vs-cats/validation", 1000)
test_features, test_labels = extract_features("./dataset/dogs-vs-cats/test", 1000)

train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

## 训练模型
model = models.Sequential()
model.add(layers.Dense(256, activation = 'relu', input_dim = 4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = optimizers.RMSprop(lr=2e-5),
              loss = 'binary_crossentropy',
              metrics = ['acc'])

history = model.fit(train_features, train_labels,
                    epochs = 30,
                    batch_size = 20,
                    validation_data = (validation_features, validation_labels))

## 绘制acc和loss曲线图
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



