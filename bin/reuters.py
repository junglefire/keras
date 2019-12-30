#!/usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import logging as log
import numpy as np
import sys
import os

from keras.utils.np_utils import to_categorical
from keras.datasets import reuters
from keras import models 
from keras import layers
from util import *

log.basicConfig(level=log.INFO)

## 设置KERAS根目录，所有的数据集都从这个目录加载
log.info("set ENV `KERAS_HOME` ...")
os.environ["KERAS_HOME"] = "./keras"

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

log.info("train datasets size: %d" % len(train_data))
log.info("test datasets size: %d" % len(test_data))

## 每个样本都是一个整数列表，表示单词索引
## 样本对应的标签是一个0~45范围内的整数，即话题索引编号
log.info("slice of a sample: %s" % train_data[10])

## 将索引解码为新闻文本
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

## Note that our indices were offset by 3
## because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

log.info("decoded newswire: %s" % decoded_newswire)

## 将训练集和测试集向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

## 对标签进行one-hot编码
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

## 构建神经网络
model = models.Sequential() 
model.add(layers.Dense(128, activation='relu', input_shape=(10000,))) 
model.add(layers.Dense(128, activation='relu')) 
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

## 留出验证集
x_val = x_train[:1000] 
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000] 
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

## 绘制训练损失和验证损失
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.subplot(211)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(212)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()








