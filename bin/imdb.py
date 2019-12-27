#!/usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import logging as log
import numpy as np
import sys
import os

from keras.datasets import imdb
from keras import models
from keras import layers
from util import *

log.basicConfig(level=log.INFO)

## 设置KERAS根目录，所有的数据集都从这个目录加载
log.info("set ENV `KERAS_HOME` ...")
os.environ["KERAS_HOME"] = "./keras"

## 加载IMDB评论数据
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

## word_index 是一个将单词映射为整数索引的字典
word_index = imdb.get_word_index()

## 键值颠倒，将整数索引映射为单词
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) 

## 将第一条评论解码
## 注意，索引减去了3， 因为0、1、2是为`padding`（填充）、`start of sequence`（序列开始）、
## `unknown`（未知词）分别保留的索引
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

log.info("the first comment: \n%s\n" % decoded_review)

## 将训练集和测试集向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

## 将标签向量化
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

## 构建模型
model = models.Sequential() 

model.add(layers.Dense(32, activation='relu', input_shape=(10000,))) 
model.add(layers.Dense(32, activation='relu')) 
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

## 留出验证集
x_val = x_train[:10000] 
partial_x_train = x_train[10000:]

y_val = y_train[:10000] 
partial_y_train = y_train[10000:]

## 训练模型
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

history_dict = history.history
log.info("history keys: %s" % history_dict.keys())

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










