#!/usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import logging as log
import numpy as np
import sys
import os

from keras.utils.np_utils import to_categorical
from keras.datasets import boston_housing
from keras import models 
from keras import layers
from util import *

log.basicConfig(level=log.INFO)

## 设置KERAS根目录，所有的数据集都从这个目录加载
log.info("set ENV `KERAS_HOME` ...")
os.environ["KERAS_HOME"] = "./keras"

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

log.info("train datasets size: %d" % len(train_data))
log.info("test datasets size: %d" % len(test_data))

## 每个样本都有13个数值特征，比如人均犯罪率、每个住宅的平均房间数、高速公路可达性等
## 目标是房屋价格的中位数，单位是千美元
log.info("shape of a train datasets: %s" % str(train_data.shape))

## 数据标准化
mean = train_data.mean(axis=0) 
train_data -= mean 
std = train_data.std(axis=0) 
train_data /= std

## 注意!!
## 用于测试数据标准化的均值和标准差都是在训练数据上计算得到的。在工作流程中， 你不能
## 使用在测试数据上计算得到的任何结果，即使是像数据标准化这么简单的事情也不行
test_data -= mean 
test_data /= std

## 因为需要将同一个模型多次实例化， 所以用一个函数来构建模型
def build_model():
	model = models.Sequential() 
	model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],))) 
	model.add(layers.Dense(64, activation='relu')) 
	model.add(layers.Dense(1))
	model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
	return model

## 通过K折验证法测试模型
k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_scores = []
all_mae_histories = []

for i in range(k):
	log.info('processing fold #%d' % i)
	# Prepare the validation data: data from partition # k
	val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
	val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
	# Prepare the training data: data from all other partitions
	partial_train_data = np.concatenate(
		[train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
		axis=0)
	partial_train_targets = np.concatenate(
		[train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]],
		axis=0)
	# Build the Keras model (already compiled)
	model = build_model()
	# Train the model (in silent mode, verbose=0)
	history = model.fit(partial_train_data, partial_train_targets,
						validation_data=(val_data, val_targets),
						epochs=num_epochs, batch_size=1, verbose=0)
	mae_history = history.history['mae']
	all_mae_histories.append(mae_history)

## 计算所有轮次中的K折验证分数平均值
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

## 绘制验证分数
# plt.subplot(211)
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history) 
plt.xlabel('Epochs') 
plt.ylabel('Validation MAE') 

## 使用移动平均值绘图
smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.subplot(212)
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history) 
plt.xlabel('Epochs') 
plt.ylabel('Validation MAE(smooth curve)') 
plt.show()




